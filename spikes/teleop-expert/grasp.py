"""Top-down grasp planner on surface points.

A grasp is: where to put the gripper's reference point (the "grip site",
between the fingertips), and the yaw angle of the line along which the two
fingers close. The gripper always points straight down.

The planner tries many (position, yaw) candidates over the object and keeps
those where
  * the open fingers come down into free space on both sides of the object,
  * the palm stays above the object and above everything next to it,
  * the pads end up level with some part of the object that is narrower than
    the gripper opening.
It then picks the candidate with the flattest pad contact, closest to the
object's middle. It works from the object's *current* shape in the world, so
it does not care whether the object stands, lies on its side or sits on top of
something else.

Gripper dimensions were measured from the MuJoCo model (Panda gripper,
robosuite 1.5.2), in the grip-site frame: x is the closing direction, z points
out of the fingertips.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Measured, metres.
HALF_OPEN = 0.0382      # pad inner face to centre line, gripper fully open
FINGER_OUT = 0.0651     # outer face of a finger to centre line, fully open
FINGER_HALF_W = 0.0105  # finger half width across the closing direction
PAD_HALF_W = 0.008
TIP_BELOW = 0.0092      # fingertip is this far beyond the grip site
PAD_BELOW = 0.0044      # pad reaches this far beyond the grip site ...
PAD_ABOVE = 0.0116      # ... and this far back toward the palm
PALM_ABOVE = 0.031      # palm underside is this far back from the grip site
HAND_HALF_L = 0.104     # palm half length along the closing direction
HAND_HALF_W = 0.0316

MARGIN = 0.004
# Horizontal distance from the robot's base axis to the grip site, fingers pointing down.
# Measured (probes/reach.py): the arm arrives within 4 mm up to 0.76-0.80 m depending on height.
MAX_REACH = 0.78
COMFORT_REACH = 0.70
NEAR_REACH = 0.47       # closer to the base than this, the arm is folded up and clumsy
MAX_WIDTH = 0.068       # widest thing we try to grasp (opening is 0.0764)
FINGER_THICK = FINGER_OUT - HALF_OPEN
NARROW_OPEN = 0.022     # half opening used where a fully open gripper does not fit


@dataclass
class Grasp:
    pos: np.ndarray     # grip-site target, world frame
    yaw: float          # angle of the closing direction about world z
    width: float        # object width between the pads
    overlap: float      # how much of the pad height touches the object
    score: float
    info: dict
    tilt: float = 0.0   # lean of the gripper about the pad width direction (see expert._top_down)
    half_open: float = HALF_OPEN  # pad-to-centre distance to approach with (smaller in tight spots)

    @property
    def u(self):
        return np.array([np.cos(self.yaw), np.sin(self.yaw)])


def site_floor(c, yaw, pts, half_gap=HALF_OPEN, margin=MARGIN):
    """Lowest grip-site height at (c, yaw) at which an open gripper touches none of `pts`."""
    if len(pts) == 0:
        return -np.inf
    u = np.array([np.cos(yaw), np.sin(yaw)])
    v = np.array([-u[1], u[0]])
    rel = pts[:, :2] - c
    pu, pv, z = np.abs(rel @ u), np.abs(rel @ v), pts[:, 2]
    lo = -np.inf
    hand = (pu <= HAND_HALF_L + margin) & (pv <= HAND_HALF_W + margin)
    if hand.any():
        lo = max(lo, z[hand].max() - PALM_ABOVE + margin)
    fing = (pu >= half_gap - 0.003) & (pu <= half_gap + (FINGER_OUT - HALF_OPEN) + margin) & (pv <= FINGER_HALF_W + margin)
    if fing.any():
        lo = max(lo, z[fing].max() + TIP_BELOW + margin)
    return lo


def _face_slopes(u, t, bin_size=0.004):
    """Slopes du/dt of the two faces the pads will touch. `u` is along the closing direction and
    `t` is a direction along the pad (across it, or up). Returns (slope of low-u face, slope of
    high-u face), or None when there is too little material to tell."""
    b = np.floor((t - t.min()) / bin_size).astype(int)
    lo, hi, tc = [], [], []
    for k in range(b.max() + 1):
        m = b == k
        if m.sum() >= 2:
            lo.append(u[m].min())
            hi.append(u[m].max())
            tc.append(t[m].mean())
    if len(tc) < 3:
        return None
    tc = np.array(tc) - np.mean(tc)
    den = float(tc @ tc)
    if den < 1e-10:
        return None
    return float(tc @ np.array(lo)) / den, float(tc @ np.array(hi)) / den


def _envelope(pts, cell=0.004):
    """Keep only the highest point in each cell of a horizontal grid. For a gripper coming
    straight down, that is all that matters about anything it must not touch."""
    if len(pts) == 0:
        return pts
    ij = np.floor(pts[:, :2] / cell).astype(np.int64)
    key = ij[:, 0] * 1_000_003 + ij[:, 1]
    order = np.lexsort((pts[:, 2], key))
    key_sorted = key[order]
    last = np.r_[key_sorted[1:] != key_sorted[:-1], True]
    return pts[order[last]]


def plan_grasps(P, Q, support_z, com, home_yaw, base_xy, n_yaw=12, v_step=0.006, u_step=0.016,
                max_obj_pts=8000, depths=(0.0, 0.006, 0.012, 0.018), rng=None, check_sweep=True,
                openings=(HALF_OPEN, NARROW_OPEN), ref_xy=None, place_r=None):
    """Return feasible top-down grasps for object points P among obstacle points Q, best first.

    For every yaw the points are rotated so that u is the closing direction and v the direction
    across it. Candidates are tried on lines of constant v. On each line the gripper centre is
    slid along u until the material at pad height is centred between the pads, at several
    heights above the lowest collision-free one.

    `ref_xy` is the object's reference point and `place_r` the distance of the place target from
    the robot base: together they let the planner drop grasps from which the arm could pick the
    object up but not reach far enough to put it down.
    """
    rng = rng or np.random.default_rng(0)
    if len(P) > max_obj_pts:
        P = P[rng.choice(len(P), max_obj_pts, replace=False)]
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    if len(Q):
        reach = HAND_HALF_L + 2 * MARGIN
        keep = (np.all(Q[:, :2] >= lo - reach, axis=1) & np.all(Q[:, :2] <= hi + reach, axis=1)
                & (Q[:, 2] > P[:, 2].min() - 0.03))
        Q = Q[keep]
    E = _envelope(np.concatenate([P, Q]) if len(Q) else P)
    QE = _envelope(Q) if len(Q) else np.zeros((0, 3))
    base_floor = support_z + TIP_BELOW + 0.003
    out = []
    yaws = home_yaw + np.linspace(-np.pi / 2, np.pi / 2, n_yaw, endpoint=False)
    for yaw in yaws:
        u = np.array([np.cos(yaw), np.sin(yaw)])
        v = np.array([-u[1], u[0]])
        Eu, Ev, Ez = E[:, :2] @ u, E[:, :2] @ v, E[:, 2]
        Pu, Pv, Pz = P[:, :2] @ u, P[:, :2] @ v, P[:, 2]
        Qu, Qv, Qz = QE[:, :2] @ u, QE[:, :2] @ v, QE[:, 2]
        for cv in np.arange(Pv.min() + 0.002, Pv.max(), v_step):
            se = np.abs(Ev - cv) <= HAND_HALF_W + MARGIN
            eu, ez = Eu[se], Ez[se]
            fe = np.abs(Ev[se] - cv) <= FINGER_HALF_W + MARGIN
            feu, fez = eu[fe], ez[fe]
            sp = np.abs(Pv - cv) <= FINGER_HALF_W
            if sp.sum() < 4:
                continue
            sq = np.abs(Qv - cv) <= FINGER_HALF_W + MARGIN
            qu, qz = Qu[sq], Qz[sq]
            pu, pv, pz = Pu[sp], Pv[sp] - cv, Pz[sp]
            on_pad_line = np.abs(pv) <= PAD_HALF_W
            for h in openings:
              done = [set() for _ in depths]
              for cu0 in np.arange(pu.min() - 0.02, pu.max() + 0.02 + u_step, u_step):
                  for k, dz in enumerate(depths):
                      cu = cu0
                      ok = False
                      for it in range(5):
                          du = np.abs(eu - cu)
                          hand = du <= HAND_HALF_L + MARGIN
                          floor = base_floor
                          if hand.any():
                              floor = max(floor, ez[hand].max() - PALM_ABOVE + MARGIN)
                          dfu = np.abs(feu - cu)
                          fing = (dfu >= h - 0.003) & (dfu <= h + FINGER_THICK + MARGIN)
                          if fing.any():
                              floor = max(floor, fez[fing].max() + TIP_BELOW + MARGIN)
                          # Nothing but the object may lie in the path the fingers sweep while closing
                          # (a drawer wall next to a bowl, say): stay above it.
                          sweep = np.abs(qu - cu) < h
                          if check_sweep and sweep.any():
                              floor = max(floor, qz[sweep].max() + TIP_BELOW + MARGIN)
                          gz = floor + dz
                          inside = np.abs(pu - cu) < h - 0.003
                          pad = inside & on_pad_line & (pz >= gz - PAD_BELOW) & (pz <= gz + PAD_ABOVE)
                          if pad.sum() < 4:
                              break
                          a, b = pu[pad].min(), pu[pad].max()
                          shift = 0.5 * (a + b) - cu
                          if abs(shift) < 0.001:
                              ok = True
                              break
                          cu = cu + shift
                          if it == 0 and round(cu / 0.002) in done[k]:
                              break
                      if not ok:
                          if k == 0 and it == 0:
                              break       # nothing at pad height at the lowest position: skip this seed
                          continue
                      key = round(cu / 0.002)
                      if key in done[k]:
                          continue
                      done[k].add(key)
                      width = b - a
                      if width > min(MAX_WIDTH, 2 * h - 0.014) or width < 0.002:
                          continue
                      overlap = min(pz[pad].max(), gz + PAD_ABOVE) - max(pz[pad].min(), gz - PAD_BELOW)
                      if overlap < 0.005:
                          continue
                      # Pad flatness: are the touched faces square to the closing direction, seen from
                      # above? Measured as the faces' slope across the pad, in metres over the pad width.
                      sl = _face_slopes(pu[pad], pv[pad])
                      slant = 0.02 if sl is None else (abs(sl[0]) + abs(sl[1])) * 2 * PAD_HALF_W
                      # How the touched faces slope with height.
                      sz = _face_slopes(pu[pad], pz[pad])
                      # A shape that narrows quickly toward the top (an upside-down bowl, the upper
                      # half of a can lying on its side) squeezes out of the pads: not a grasp.
                      low_face = inside & (pz >= gz - TIP_BELOW) & (pz <= gz + PAD_ABOVE)
                      sw = _face_slopes(pu[low_face], pz[low_face])
                      if sw is not None and sw[0] - sw[1] > 1.0:
                          continue
                      # Lean: a thin wall that leans (a bowl's rim) has both faces sloping the same way.
                      # Flat pads squeeze such a wall out unless the gripper leans with it (measured:
                      # 71% of rim grasps survive a shake without the lean, 88-97% with it).
                      tilt, lean_pen = 0.0, 0.0
                      if width < 0.02:
                          if sz is None or abs(sz[0] - sz[1]) >= 0.15:
                              lean_pen = 0.04      # thin, and too little of it between the pads to tell
                          elif abs(0.5 * (sz[0] + sz[1])) > 0.08:
                              full = float(np.clip(np.arctan(0.5 * (sz[0] + sz[1])), -0.45, 0.45))
                              side = np.sign(full)
                              x = eu[hand] - cu
                              low = ((feu - cu) * side >= h - 0.003) & ((feu - cu) * side <= h + FINGER_THICK + MARGIN)
                              under = max(support_z, float(fez[low].max()) if low.any() else -np.inf)
                              # Leaning lowers one end of the long palm and one fingertip. Use the
                              # largest part of the full lean for which both still clear everything.
                              for frac in (1.0, 0.75, 0.5, 0.33):
                                  t = full * frac
                                  need_palm = float((ez[hand] + MARGIN - PALM_ABOVE * np.cos(t) + x * np.sin(t)).max())
                                  need_tip = under + TIP_BELOW + (h + FINGER_THICK) * abs(np.sin(t)) + MARGIN
                                  if max(need_palm, need_tip) <= gz + 0.001:
                                      tilt = t
                                      break
                              lean_pen = 0.04 * (1.0 - abs(tilt / full))
                      # Bulge: how much wider the object is along the rest of the finger than at the
                      # pads. A bulge means the smooth finger body, not the pad, touches first.
                      face = inside & (pz >= gz - TIP_BELOW) & (pz <= gz + PALM_ABOVE)
                      bulge = 0.0 if tilt else max(0.0, (pu[face].max() - pu[face].min()) - width)
                      depth = float(pz[inside].max() - gz)
                      cc = cu * u + cv * v
                      arm = float(np.linalg.norm(cc - com[:2]))
                      dyaw = abs(yaw - home_yaw)
                      reach_d = float(np.linalg.norm(cc - base_xy))
                      if reach_d > MAX_REACH:
                          continue
                      # Where will the gripper be when the object is over the target? It keeps its
                      # offset from the object's reference point (4.7 cm for a bowl held by the rim).
                      place_d = 0.0
                      if place_r is not None and ref_xy is not None:
                          off = cc - ref_xy
                          place_d = place_r + float(off @ (cc - base_xy)) / max(reach_d, 1e-6)
                          if place_d > MAX_REACH - 0.01:
                              continue
                      score = (4.0 * slant + 3.0 * bulge + 0.6 * arm + 0.004 * dyaw + 0.02 * reach_d
                               - 1.0 * min(overlap, 0.016) - 0.1 * min(depth, 0.03) + 0.3 * max(0.0, width - 0.06)
                               + 1.5 * max(0.0, reach_d - COMFORT_REACH) + 1.5 * max(0.0, place_d - COMFORT_REACH)
                               + 4.0 * max(0.0, NEAR_REACH - reach_d))
                      if tilt:
                          # For a leaning wall, measured: grips 14 to 22 mm below the rim hold best.
                          score += 0.5 * abs(depth - 0.018)
                          # Leaning the gripper sideways (about the line from the robot to the grasp)
                          # throws the wrist around when the arm is folded close to its base; leaning
                          # it forward or back does not. So prefer a closing direction that points
                          # along that line, the more so the closer the grasp is to the robot.
                          radial = (cc - base_xy) / max(reach_d, 1e-6)
                          sideways = abs(u[0] * radial[1] - u[1] * radial[0])
                          score += sideways * (0.01 + 0.2 * max(0.0, 0.55 - reach_d))
                      score += lean_pen
                      if h < HALF_OPEN:
                          score += 0.01
                      out.append(Grasp(np.array([cc[0], cc[1], gz]), float(yaw), float(width), float(overlap),
                                       float(score), dict(slant=float(slant), bulge=float(bulge), arm=arm, depth=depth),
                                       tilt=tilt, half_open=h))
    out.sort(key=lambda g: g.score)
    return out
