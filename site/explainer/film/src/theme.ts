// The film's visual language: one place for colour, type, space and motion.
// Bright and calm: warm paper, ink type, colour only where it means something.

export const W = 1920;
export const H = 1080;
export const FPS = 30;

export const color = {
  paper: '#F5F2EA',
  paper2: '#ECE7DC', // a band or a panel on paper
  card: '#FFFFFF',
  ink: '#15171C',
  ink2: '#5A606E', // secondary text
  ink3: '#8E929C', // labels, axes
  line: '#DCD5C6', // rules, empty dots
  ok: '#12A26C', // delivered, success
  bad: '#E5484D', // failed
  agent: '#4B5CF0', // the engineering agent
  lead: '#E08A1E', // the shift lead, a request
  teal: '#0F8B8D', // checks
  violet: '#9A6BEA', // story B's second cause
  wood: '#B98E5E', // the library shelf
  night: '#15171C', // dark cards (agent's notes, the vault)
  nightText: '#C9CBD3',
} as const;

export const font = {
  sans: '"Inter Variable", Inter, "Helvetica Neue", Arial, sans-serif',
  mono: '"JetBrains Mono Variable", "JetBrains Mono", ui-monospace, Menlo, monospace',
} as const;

// Type scale at 1920x1080.
export const type = {
  kicker: {fontSize: 26, fontWeight: 750, letterSpacing: '0.14em', textTransform: 'uppercase' as const, color: color.ink3},
  display: {fontSize: 132, fontWeight: 820, letterSpacing: '-0.035em', lineHeight: 0.98, color: color.ink},
  h1: {fontSize: 92, fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1.02, color: color.ink},
  h2: {fontSize: 64, fontWeight: 790, letterSpacing: '-0.02em', lineHeight: 1.05, color: color.ink},
  h3: {fontSize: 44, fontWeight: 760, letterSpacing: '-0.012em', lineHeight: 1.12, color: color.ink},
  body: {fontSize: 34, fontWeight: 480, lineHeight: 1.35, color: color.ink2},
  small: {fontSize: 26, fontWeight: 520, lineHeight: 1.35, color: color.ink2},
  mono: {fontFamily: '"JetBrains Mono Variable", ui-monospace, monospace', fontSize: 28, lineHeight: 1.45},
} as const;

export const space = {
  marginX: 120, // safe margin left and right
  marginY: 90, // safe margin top and bottom
  gap: 32,
  radius: 24, // cards
  radiusFrame: 28, // footage frames
} as const;

export const shadow = {
  card: '0 14px 44px rgba(20,22,28,0.12)',
  frame: '0 24px 70px rgba(20,22,28,0.22)',
  pill: '0 6px 20px rgba(20,22,28,0.16)',
} as const;

// Motion: purposeful and calm. Springs for arrivals, easing for camera-like moves.
export const spring = {
  calm: {damping: 200, mass: 1, stiffness: 120}, // no overshoot
  lively: {damping: 14, mass: 0.8, stiffness: 140}, // a small overshoot for pops
} as const;

export const durations = {
  enter: 16, // frames for an element to arrive
  stagger: 4, // frames between items arriving in sequence
  dotStagger: 2, // frames between dots filling
  transition: 14, // frames of a cut transition between beats
} as const;
