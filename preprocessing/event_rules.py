"""Sportec(IDSSE) 소유 라벨을 대회 정의("제어된 소유")에 맞게 고치는 스크립트.

대회 정의 (CHALLENGE.md)
  - 상대 팀이 공을 '제어해서' 가져가야 소유가 바뀐다.
  - 단순 터치, 패스를 끊지 못한 굴절, 제어로 이어지지 않은 공중볼 승리는 소유 변경이 아니다.
  - 공이 경합 중이거나 주인이 없으면 마지막으로 가졌던 팀의 소유가 유지된다.

방법
  1. 위치 XML 의 공(BALL) 프레임에서 원래 라벨을 읽는다: BallStatus(1 = alive), BallPossession(1 또는 2).
  2. 이벤트 XML 의 각 이벤트 시각(EventTime)을 프레임 번호로 바꾼다 (하프 첫 프레임의 시각·번호 기준, 25 fps).
  3. 원래 라벨로 소유 구간(같은 팀이 이어지는 alive 프레임 묶음, dead 나 하프에서 끊김)을 나눈다.
  4. 구간마다 그 팀의 '제어 증거' 이벤트가 구간 안(앞뒤 TOL 프레임 여유 포함)에 있는지 본다.
  5. 증거가 없으면 그 구간을 직전 구간의 팀 소유로 바꾼다 (경합·단순 터치로 봄).
     dead 직후 첫 구간은 증거가 없어도 그대로 둔다 (재개 때는 소유자가 분명함).

제어 증거 (기준별)
  strict(엄격): 성공 패스, 공 확보·가로채기(BallClaiming), 슛, 드리블(Run),
                경합 승리 중 승자가 공을 제어했거나(WinnerRole=withBallControl) 소유가 바뀐 것으로 기록된 것(PossessionChange=true)
  mid(중간)   : strict + 실패한 패스 (패스를 시도했다 = 공을 발밑에 갖고 있었다)
  loose(느슨) : mid + 기타 볼 액션(OtherBallAction)

사용
  python relabel_controlled_possession.py --raw <Sportec raw 폴더> --out <출력 폴더>
  <raw> 아래: DFL_04_03_positions_raw_observed_*.xml, event/DFL_03_02_events_raw_*.xml
출력: <out>/<경기>_labels.csv  (25 fps)  frame_id, period, alive, poss_orig, poss_loose, poss_mid, poss_strict
      poss 값: 1 또는 2 = BallPossession 의 팀 번호, 0 = dead
"""
import argparse, glob, os, re
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd

FPS = 25
TOL = 12   # 이벤트 시각과 프레임의 어긋남 여유 (프레임, 약 0.5초)
STRICT = ["play_ok", "claim_BallClaimed", "claim_BallHeld", "claim_InterceptedBall", "shot", "duel_ctrl", "run"]
LEVELS = {"strict": STRICT, "mid": STRICT + ["play_fail"], "loose": STRICT + ["play_fail", "other"]}
ATTR = re.compile(r'(\w+)="([^"]*)"')


def read_ball(pos_xml):
    """위치 XML 에서 공 프레임만 읽음 -> frame_id, period, alive, poss, 하프별 (첫 프레임 번호, 시각)"""
    rows, halves, inball, per = [], {}, False, 0
    with open(pos_xml, encoding="utf-8") as f:
        for line in f:
            if "<FrameSet" in line:
                inball = 'TeamId="BALL"' in line
                if inball:
                    per = 1 if "firstHalf" in line else 2
                continue
            if inball and "<Frame " in line:
                a = dict(ATTR.findall(line))
                n = int(a["N"])
                if per not in halves:
                    halves[per] = (n, pd.Timestamp(a["T"]))
                rows.append((n, per, a.get("BallStatus") == "1", int(a.get("BallPossession", 0))))
    df = pd.DataFrame(rows, columns=["frame_id", "period", "alive", "poss"]).sort_values("frame_id").reset_index(drop=True)
    return df, halves


def read_events(ev_xml, halves):
    """이벤트 -> (프레임 번호, 팀 ID, 종류)"""
    out = []
    h2 = halves[2]
    for e in ET.parse(ev_xml).getroot().iter("Event"):
        t = pd.Timestamp(e.get("EventTime"))
        n0, t0 = h2 if t >= h2[1] - pd.Timedelta(seconds=5) else halves[1]
        f = int(round(n0 + (t - t0).total_seconds() * FPS))
        for c in e.iter():
            g = c.tag
            if g == "Play":
                out.append((f, c.get("Team"), "play_ok" if c.get("Evaluation") == "successfullyCompleted" else "play_fail"))
            elif g == "BallClaiming":
                out.append((f, c.get("Team"), "claim_" + str(c.get("Type"))))
            elif g == "ShotAtGoal":
                out.append((f, c.get("Team"), "shot"))
            elif g == "TacklingGame":
                ctrl = c.get("WinnerRole") == "withBallControl" or c.get("PossessionChange") == "true"
                out.append((f, c.get("WinnerTeam"), "duel_ctrl" if ctrl else "duel_noctrl"))
            elif g == "OtherBallAction":
                out.append((f, c.get("Team"), "other"))
            elif g == "Run":
                out.append((f, c.get("Team"), "run"))
    ev = pd.DataFrame(out, columns=["f", "team", "kind"])
    return ev[ev.team.notna()]


def spells(lab, period):
    """같은 값이 이어지는 구간 (시작, 끝). 하프가 바뀌면 끊음."""
    brk = np.r_[True, (lab[1:] != lab[:-1]) | (period[1:] != period[:-1]), True]
    ch = np.flatnonzero(brk)
    return [(ch[i], ch[i + 1] - 1) for i in range(len(ch) - 1)]


def relabel(lab, period, ev_i, ev_code, ev_kind, kinds):
    use = np.isin(ev_kind, kinds); ei, ec = ev_i[use], ev_code[use]
    new = lab.copy(); prev = None
    for s, t in spells(lab, period):
        if lab[s] == 0:                       # dead: 다음 구간은 재개 -> 그대로 둠
            prev = None; continue
        if prev is not None and period[s] != period[s - 1]:
            prev = None
        has = ((ei >= s - TOL) & (ei <= t + TOL) & (ec == lab[s])).any()
        if prev is not None and not has:
            new[s:t + 1] = prev               # 제어 증거 없음 -> 직전 팀 소유 유지
        else:
            prev = lab[s]
    return new


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True); ap.add_argument("--match", nargs="*")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for pos in sorted(glob.glob(os.path.join(a.raw, "DFL_04_03_positions_raw_observed_*.xml"))):
        m = re.search(r"MAT-(\w+)\.xml$", pos).group(1)
        if a.match and m not in a.match:
            continue
        ball, halves = read_ball(pos)
        ev = read_events(glob.glob(os.path.join(a.raw, "event", f"*{m}.xml"))[0], halves)
        fr = ball.frame_id.to_numpy(); per = ball.period.to_numpy(); alive = ball.alive.to_numpy(); poss = ball.poss.to_numpy()
        i = np.searchsorted(fr, ev.f.to_numpy()).clip(0, len(fr) - 1)
        # 이벤트의 팀 ID -> BallPossession 번호(1/2): 성공 패스 시점의 원래 소유 번호 중 다수
        ok = (ev.kind.to_numpy() == "play_ok") & alive[i]
        ct = pd.crosstab(ev.team.to_numpy()[ok], poss[i][ok]); code = {t: int(ct.loc[t].idxmax()) for t in ct.index}
        agree = np.mean([code[t] == p for t, p in zip(ev.team.to_numpy()[ok], poss[i][ok])])
        ev_code = ev.team.map(code).fillna(-1).to_numpy().astype(int)
        lab = np.where(alive, poss, 0)
        out = pd.DataFrame({"frame_id": fr, "period": per, "alive": alive.astype(int), "poss_orig": lab})
        msg = [f"{m}: 이벤트 {len(ev)}개, 성공 패스 팀 일치 {agree:.3f}"]
        for name, kinds in LEVELS.items():
            new = relabel(lab, per, i, ev_code, ev.kind.to_numpy(), kinds); out["poss_" + name] = new
            sp = [(s, t) for s, t in spells(new, per) if new[s] != 0]; dur = np.array([t - s + 1 for s, t in sp]) / FPS
            msg.append(f"{name}: 소유 {len(sp)}개 평균 {dur.mean():.1f}초, 바뀐 alive 프레임 {(new != lab)[alive].mean():.3f}")
        sp = [(s, t) for s, t in spells(lab, per) if lab[s] != 0]; dur = np.array([t - s + 1 for s, t in sp]) / FPS
        msg.insert(1, f"원래: 소유 {len(sp)}개 평균 {dur.mean():.1f}초")
        out.to_csv(os.path.join(a.out, f"{m}_labels.csv"), index=False)
        print(" | ".join(msg), flush=True)


if __name__ == "__main__":
    main()
