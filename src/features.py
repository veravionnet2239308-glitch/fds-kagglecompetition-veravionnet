import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_features(data):
    rows = []
    for b in tqdm(data, desc="Extracting features"):
        r = {"battle_id": b.get("battle_id")}

        if "player_won" in b:
            r["player_won"] = int(b["player_won"])

        p1 = b.get("p1_team_details", []) or []
        p2lead = b.get("p2_lead_details", {}) or {}

        for stat in ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']:
            vals = [p.get(stat, 0) for p in p1]
            if not vals:
                vals = [0]
            r[f"p1_mean_{stat}"] = float(np.mean(vals))
            r[f"p2lead_{stat}"]  = p2lead.get(stat, 0)
            r[f"delta_{stat}"]   = r[f"p1_mean_{stat}"] - r[f"p2lead_{stat}"]

        r["p1_offense"] = r["p1_mean_base_atk"] + r["p1_mean_base_spa"]
        r["p1_defense"] = r["p1_mean_base_def"] + r["p1_mean_base_spd"]
        r["p2_offense"] = r["p2lead_base_atk"] + r["p2lead_base_spa"]
        r["p2_defense"] = r["p2lead_base_def"] + r["p2lead_base_spd"]
        r["offense_diff"] = r["p1_offense"] - r["p2_offense"]
        r["defense_diff"] = r["p1_defense"] - r["p2_defense"]

        tl = b.get("battle_timeline", [])[:10]

        p1_hp, p2_hp = [], []
        p1_status = p2_status = 0
        p1_first_strike = p2_first_strike = 0
        damage_turn_p1, damage_turn_p2 = [], []
        dominance_p1 = dominance_p2 = 0
        prev_hp1 = prev_hp2 = None

        for t in tl:
            p1s = t.get("p1_pokemon_state", {}) or {}
            p2s = t.get("p2_pokemon_state", {}) or {}
            p1m = t.get("p1_move_details")
            p2m = t.get("p2_move_details")

            hp1 = p1s.get("hp_pct")
            hp2 = p2s.get("hp_pct")

            if hp1 is not None:
                p1_hp.append(hp1)
                if prev_hp1 is not None:
                    damage_turn_p1.append(max(0, prev_hp1 - hp1))
                prev_hp1 = hp1

            if hp2 is not None:
                p2_hp.append(hp2)
                if prev_hp2 is not None:
                    damage_turn_p2.append(max(0, prev_hp2 - hp2))
                prev_hp2 = hp2

            if p1s.get("status", "nostatus") != "nostatus":
                p1_status += 1
            if p2s.get("status", "nostatus") != "nostatus":
                p2_status += 1

            if p1m and p2m:
                p1_pri = p1m.get("priority", 0)
                p2_pri = p2m.get("priority", 0)
                if p1_pri > p2_pri:
                    p1_first_strike += 1
                elif p2_pri > p1_pri:
                    p2_first_strike += 1
                else:
                    if r["p1_mean_base_spe"] > r["p2lead_base_spe"]:
                        p1_first_strike += 1
                    elif r["p1_mean_base_spe"] < r["p2lead_base_spe"]:
                        p2_first_strike += 1
            elif p1m:
                p1_first_strike += 1
            elif p2m:
                p2_first_strike += 1

            if hp1 == 0:
                pass
            if hp2 == 0:
                pass

            if (hp1 is not None) and (hp2 is not None):
                if hp1 > hp2:
                    dominance_p1 += 1
                elif hp2 > hp1:
                    dominance_p2 += 1

        r["p1_hp_avg"] = float(np.mean(p1_hp)) if p1_hp else 1.0
        r["p2_hp_avg"] = float(np.mean(p2_hp)) if p2_hp else 1.0
        r["hp_diff"] = r["p1_hp_avg"] - r["p2_hp_avg"]

        r["p1_status_count"] = p1_status
        r["p2_status_count"] = p2_status
        r["status_diff"] = p1_status - p2_status

        r["p1_first_strike"] = p1_first_strike
        r["p2_first_strike"] = p2_first_strike
        r["first_strike_diff"] = p1_first_strike - p2_first_strike

        r["avg_damage_p1"] = float(np.mean(damage_turn_p1)) if damage_turn_p1 else 0.0
        r["avg_damage_p2"] = float(np.mean(damage_turn_p2)) if damage_turn_p2 else 0.0
        r["avg_damage_diff"] = r["avg_damage_p2"] - r["avg_damage_p1"]

        r["dominance_p1"] = dominance_p1
        r["dominance_p2"] = dominance_p2
        r["dominance_diff"] = dominance_p1 - dominance_p2

        rows.append(r)

    return pd.DataFrame(rows).fillna(0.0)
