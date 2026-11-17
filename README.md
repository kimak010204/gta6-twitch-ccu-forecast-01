# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kimak010204/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kimak010204/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast Grand Theft Auto VI's launch-day Twitch viewers hour by hour, from the first hour's count. `main.py` (Python 3.13, standard library only) reads each case in `/input` and writes five quantiles (5%, 25%, 50%, 75%, 95%) for hours 2–24 to `/output`. The model is a per-hour ridge regression over the past big launches shipped in the case itself: the log-ratio of each hour's viewers to hour 1 is predicted from the size of hour 1 (log scale, quadratic), whether the launch unlocked at rolling midnight local, and step features for the US East/West midnight unlocks. The spread comes from the residuals of the rolling, non-early-access launches, shrunk halfway toward a normal spread, and is asymmetric in hours 2–16 (full-width lower tail, halved upper tail) to cover sagging launches like Spider-Man 2 without overstating the midday upside; far outside the fitted hour-1 range the spread widens as an extrapolation hedge. Practice mean pinball loss 0.122, worst case 0.168 (house baseline 0.387).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=t87hzjc5&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
