-- How late do each robot's commands take effect? From the visible tables `steps` and
-- `episodes` only (DuckDB, as the agents' query tool runs it).
--
-- For every lag L from 0 to 30 control steps: the mean distance between the end-effector
-- position commanded L steps before the step that produced a measurement and the position
-- measured, over the steps from 41 on (the same steps for every L). The best lag per
-- robot is the arm controller's tracking lag (3 steps on the healthy calibration runs)
-- plus any delay the commands meet on their way to the robot. Commanded positions are in
-- the world frame under absolute control, like the measured ones.
WITH s AS (
  SELECT st.episode_id, e.robot_id, st.step, st.eef_pos AS m, st.commanded_action AS c
  FROM steps AS st
  JOIN episodes AS e USING (episode_id)
),
lags AS (SELECT range AS lag FROM range(0, 31)),
err AS (
  SELECT now.robot_id, lags.lag,
         avg(sqrt((cmd.c[1] - now.m[1]) ^ 2 + (cmd.c[2] - now.m[2]) ^ 2
                  + (cmd.c[3] - now.m[3]) ^ 2)) AS dist
  FROM s AS now
  CROSS JOIN lags
  JOIN s AS cmd ON cmd.episode_id = now.episode_id AND cmd.step = now.step - 1 - lags.lag
  WHERE now.step >= 41 AND cmd.c IS NOT NULL
  GROUP BY now.robot_id, lags.lag
)
SELECT robot_id, arg_min(lag, dist) AS lag_steps, round(min(dist), 4) AS distance_m
FROM err
GROUP BY robot_id
ORDER BY robot_id;
