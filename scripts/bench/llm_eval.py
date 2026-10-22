import os
import sys

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import base64
import hashlib
import html
import json
import re
import shutil
import time

import numpy as np

import generic_eval as ge
from sarge import scene as scenes
from sarge import writer as wr
from sarge.generators.routed import side_for
from sarge.prompts import build, compose, template, template_slots
from sarge.types import AudioClip, CacheKey
from sarge.world import bucket

ROOT = ge.ROOT
SCRIPTS = ge.SCRIPTS
GATE = os.path.join(ROOT, "data", "_gate", "llm")
SCENES = os.path.join(GATE, "scenes40.json")
WORK = os.path.join(GATE, "_work")
WATER_DIR = os.path.join(GATE, "_water")
PAGE = os.path.join(GATE, "page")
BASE_ARMS = ("phrasebook", "caption", "llm")
VARIANTS = ("llm_nosound", "llm_named")
STYLE_ARM = {s: "llm" if s == wr.DEFAULT_STYLE else "wr_" + s for s in wr.STYLES}
STYLE_ARMS = tuple(a for a in STYLE_ARM.values() if a != "llm")
GROUND = {"rg_anchor": "anchors", "rg_words": "words", "rg_caption": "caption"}
GROUND_ARMS = tuple(GROUND)
GROUND_BASE = {"anchors": "named"}
EXTRA_ARMS = ("ceiling",)
FLOW_BASE = {"fl_phrasebook": "phrasebook", "fl_caption": "caption"}
FLOW_ARMS = tuple(FLOW_BASE)
ENGINES = {"dflow": "woosh", "flow": "woosh-flow", "flow_rk4_8": "woosh-flow-rk4-8", "flow_rk4_16": "woosh-flow-rk4-16",
           "flow_mid_16": "woosh-flow-mid-16", "flow_mid_32": "woosh-flow-mid-32", "dflow_6": "woosh-6",
           "dflow_8": "woosh-8"}
ENGINE_SPEC = {"dflow": ("dflow", 4, "dopri5"), "flow": ("flow", 4, "dopri5"), "flow_rk4_8": ("flow", 8, "rk4"),
               "flow_rk4_16": ("flow", 16, "rk4"), "flow_mid_16": ("flow", 16, "midpoint"),
               "flow_mid_32": ("flow", 32, "midpoint"), "dflow_6": ("dflow", 6, "dopri5"),
               "dflow_8": ("dflow", 8, "dopri5")}
SPEED_BASE = {"fc_rk4_8": ("caption", "flow_rk4_8"), "fc_rk4_16": ("caption", "flow_rk4_16"),
              "fc_mid_16": ("caption", "flow_mid_16"), "fc_mid_32": ("caption", "flow_mid_32"),
              "dc_6": ("caption", "dflow_6"), "dc_8": ("caption", "dflow_8"), "dp_8": ("phrasebook", "dflow_8")}
SPEED_ARMS = tuple(SPEED_BASE)
MODEL_NAME = dict(ge.MODEL_NAME, **{"woosh-flow": "Woosh Flow", "woosh-flow-rk4-8": "Woosh Flow, rk4 8 steps",
                                    "woosh-flow-rk4-16": "Woosh Flow, rk4 16 steps",
                                    "woosh-flow-mid-16": "Woosh Flow, midpoint 16 steps",
                                    "woosh-flow-mid-32": "Woosh Flow, midpoint 32 steps",
                                    "woosh-6": "Woosh DFlow, 6 steps", "woosh-8": "Woosh DFlow, 8 steps"})
ALL_ARMS = BASE_ARMS + VARIANTS + STYLE_ARMS + GROUND_ARMS + EXTRA_ARMS + FLOW_ARMS + SPEED_ARMS
ARMS = BASE_ARMS
SETS = ("known", "fresh", "all", "fresh3", "all3", "dev")
FRESH = (
    dict(id="handgun_fire", event_type="handgun_fire", material="glock", type="gunshot_pistol"),
    dict(id="carbine_shot", event_type="carbine_shot", material="carbine", type="gunshot_rifle"),
    dict(id="gunfire_revolver", event_type="gunfire", material="", type="gunshot_pistol",
         scene={"src": "npc", "who": "Bandit_Gunslinger", "obj": "Revolver_Colt", "act": "use", "room": "open"}),
    dict(id="shotgun_shot", event_type="shotgun_shot", material="", type="gunshot_shotgun",
         scene={"src": "player", "obj": "Shotgun_Pump", "act": "use", "room": "open"}),
    dict(id="hunting_rifle_shot", event_type="hunting_rifle_shot", material="", type="gunshot_rifle",
         scene={"src": "player", "obj": "Rifle_Hunting_Bolt", "act": "use", "room": "open"}),
    dict(id="rocket_explosion", event_type="rocket_explosion", material="missile", type="explosion",
         approx="fireworks stand in for a rocket explosion"),
    dict(id="detonate_tank", event_type="detonate", material="", intensity="hard", type="explosion",
         approx="fireworks stand in for a gas tank detonation",
         scene={"src": "world", "obj": "Gas_Tank_Propane", "act": "break", "room": "open"}),
    dict(id="footstep_mud", event_type="Footstep_Mud", material="mud", type="footstep"),
    dict(id="walk_carpet", event_type="walk_carpet", material="carpet", type="footstep"),
    dict(id="step_marble", event_type="step", material="marble", type="footstep",
         scene={"src": "player", "act": "step", "room": "medium"}),
    dict(id="npc_footstep_sand", event_type="npc_footstep", material="sand", type="footstep",
         scene={"src": "npc", "who": "Guard_Patrol", "act": "step", "room": "open"}),
    dict(id="jar_smash", event_type="jar_smash", material="glass", type="glass_break"),
    dict(id="beer_bottle_break", event_type="beer_bottle_break", material="glass", type="glass_break"),
    dict(id="mirror_break", event_type="mirror_break", material="glass", type="glass_break",
         scene={"src": "projectile", "obj": "Mirror_Wall", "act": "break", "room": "small"}),
    dict(id="barn_door_open", event_type="barn_door_open", material="wood", type="door_open"),
    dict(id="swing_open_cellar", event_type="swing_open", material="", type="door_open",
         scene={"src": "player", "obj": "Door_Cellar", "act": "use", "room": "small"}),
    dict(id="locker_slam", event_type="locker_slam", material="metal", type="door_slam",
         approx="house door slams stand in for a locker door"),
    dict(id="door_shut_bedroom", event_type="door_shut", material="wood", intensity="hard", type="door_slam",
         scene={"src": "npc", "who": "Innkeeper", "obj": "Door_Bedroom", "act": "use", "room": "medium"}),
    dict(id="guard_dog_bark", event_type="guard_dog_bark", material="dog", type="dog_bark"),
    dict(id="woof_shepherd", event_type="woof", material="", type="dog_bark",
         scene={"src": "npc", "who": "Dog_Shepherd", "act": "use", "room": "open"}),
    dict(id="uppercut", event_type="uppercut", material="jaw", type="punch",
         approx="foley punches stand in for an uppercut"),
    dict(id="brawl_punch", event_type="brawl_punch", material="body", type="punch",
         scene={"src": "npc", "who": "Bar_Brawler", "obj": "Fist_L", "act": "hit", "room": "medium"}),
    dict(id="switch_flip", event_type="switch_flip", material="plastic", type="ui_click",
         approx="mouse and button clicks stand in for a wall switch"),
    dict(id="key_press_keypad", event_type="key_press", material="plastic", type="ui_click",
         scene={"src": "player", "obj": "Keypad_Security", "act": "use", "room": "small"}),
    dict(id="bus_horn", event_type="bus_horn", material="bus", type="horn", approx="car horns stand in for a bus horn"),
    dict(id="honk_van", event_type="honk", material="", type="horn", approx="car horns stand in for a van horn",
         scene={"src": "npc", "who": "Delivery_Driver", "obj": "Van_Delivery", "act": "use", "room": "open"}),
    dict(id="parry_katana", event_type="parry", material="steel", type="sword",
         approx="sword clashes and draws (the sword set)",
         scene={"src": "npc", "who": "Samurai_Duelist", "obj": "Katana", "act": "hit", "room": "open"}),
    dict(id="dagger_clash", event_type="dagger_clash", material="iron", type="sword",
         approx="sword clashes and draws (the sword set)"),
    dict(id="dive_splash", event_type="dive_splash", material="water", type="splash", water=True),
    dict(id="plop_pond", event_type="plop", material="water", type="splash", water=True,
         scene={"src": "world", "obj": "Pond_Lily", "act": "hit", "room": "open"}),
)
FRESH3 = (
    dict(id="pistol_gunshot", event_type="pistol_gunshot", material="", type="gunshot_pistol"),
    dict(id="sidearm_fire", event_type="sidearm_fire", material="", type="gunshot_pistol",
         scene={"src": "npc", "who": "Security_Guard", "obj": "Sidearm_Compact", "act": "use", "room": "medium"}),
    dict(id="assault_rifle_shot", event_type="assault_rifle_shot", material="", type="gunshot_rifle",
         scene={"src": "player", "obj": "Rifle_Assault", "act": "use", "room": "open"}),
    dict(id="marksman_fire", event_type="marksman_fire", material="rifle", type="gunshot_rifle"),
    dict(id="buckshot_fire", event_type="buckshot_fire", material="shotgun", type="gunshot_shotgun"),
    dict(id="artillery_explosion", event_type="artillery_explosion", material="shell", intensity="hard",
         type="explosion", approx="fireworks stand in for an artillery shell"),
    dict(id="demolition_blast", event_type="demolition_blast", material="", type="explosion",
         approx="fireworks stand in for a demolition charge",
         scene={"src": "world", "obj": "Charge_Demolition", "act": "break", "room": "open"}),
    dict(id="jog_leaves", event_type="jog_leaves", material="leaves", type="footstep"),
    dict(id="tread_wood", event_type="tread", material="wood", type="footstep",
         scene={"src": "player", "act": "step", "room": "small"}),
    dict(id="npc_walk_stone", event_type="npc_walk", material="stone", type="footstep",
         scene={"src": "npc", "who": "Villager_Old", "act": "step", "room": "corridor"}),
    dict(id="sprint_asphalt", event_type="sprint_asphalt", material="asphalt", type="footstep"),
    dict(id="vase_break", event_type="vase_break", material="glass", type="glass_break"),
    dict(id="pane_smash", event_type="pane_smash", material="glass", type="glass_break",
         scene={"src": "projectile", "obj": "Window_Shop", "act": "break", "room": "open"}),
    dict(id="goblet_break", event_type="goblet_break", material="glass", type="glass_break",
         scene={"src": "npc", "who": "Tavern_Brawler", "obj": "Goblet", "act": "break", "room": "medium"}),
    dict(id="gate_open", event_type="gate_open", material="wood", type="door_open"),
    dict(id="hatch_open", event_type="hatch_open", material="", type="door_open",
         scene={"src": "player", "obj": "Hatch_Trapdoor", "act": "use", "room": "small"}),
    dict(id="cabinet_slam", event_type="cabinet_slam", material="wood", type="door_slam",
         approx="house door slams stand in for a cabinet door"),
    dict(id="slam_shut", event_type="slam_shut", material="", intensity="hard", type="door_slam",
         scene={"src": "npc", "who": "Shopkeeper", "obj": "Door_Shop", "act": "use", "room": "medium"}),
    dict(id="hound_bark", event_type="hound_bark", material="", type="dog_bark"),
    dict(id="yap_terrier", event_type="yap", material="", type="dog_bark",
         scene={"src": "npc", "who": "Dog_Terrier", "act": "use", "room": "open"}),
    dict(id="jab", event_type="jab", material="face", type="punch"),
    dict(id="fist_strike", event_type="fist_strike", material="body", type="punch",
         scene={"src": "player", "obj": "Fist_R", "act": "hit", "room": "small"}),
    dict(id="toggle_click", event_type="toggle_click", material="plastic", type="ui_click",
         approx="mouse and button clicks stand in for a toggle switch"),
    dict(id="remote_press", event_type="remote_press", material="", type="ui_click",
         approx="mouse and button clicks stand in for a remote control button",
         scene={"src": "player", "obj": "Remote_Control", "act": "use", "room": "small"}),
    dict(id="motorcycle_horn", event_type="motorcycle_horn", material="motorcycle", type="horn",
         approx="car horns stand in for a motorcycle horn"),
    dict(id="horn_beep_cab", event_type="horn_beep", material="", type="horn",
         approx="car horns stand in for a sedan horn",
         scene={"src": "npc", "who": "Cab_Driver", "obj": "Car_Sedan", "act": "use", "room": "open"}),
    dict(id="saber_clash", event_type="saber_clash", material="steel", type="sword",
         approx="sword clashes and draws (the sword set)"),
    dict(id="swordfight_hit", event_type="swordfight_hit", material="", type="sword",
         approx="sword clashes and draws (the sword set)",
         scene={"src": "npc", "who": "Knight_Guard", "obj": "Longsword", "act": "hit", "room": "open"}),
    dict(id="belly_flop", event_type="belly_flop", material="water", type="splash", water=True),
    dict(id="splashdown_crate", event_type="splashdown", material="water", type="splash", water=True,
         scene={"src": "world", "obj": "Crate_Wooden", "act": "hit", "room": "open"}),
)
CAPTION = "template:caption+tt"
FIRST, SEEDS, TAKES = 40, 20, 4
AMP = "full"
ROUTE = "families"
BENCH_TYPE = {"pistol_shot": "gunshot_pistol", "rifle_shot": "gunshot_rifle", "step_stone": "footstep",
              "step_gravel": "footstep", "step_wood": "footstep", "step_metal": "footstep", "step_grass": "footstep",
              "step_water": "footstep", "iron_boots_step": "footstep", "bottle_shot_pistol": "glass_break",
              "window_rifle": "glass_break", "punch": "punch", "car_horn": "horn", "door_creak": "door_open",
              "dog_bark": "dog_bark", "sword_clash": "sword", "keyboard_typing": "ui_click", "explosion": "explosion",
              "snow_step": "footstep", "glass_shatter": "glass_break", "car_door_slam": "door_slam"}
BENCH_APPROX = {"bottle_shot_pistol": "glass-break references for a bullet through a bottle",
                "keyboard_typing": "mouse, button and keyboard clicks stand in for typing",
                "explosion": "fireworks stand in for an exploding barrel",
                "car_door_slam": "house door slams stand in for a car door",
                "glass_shatter": "window and glass breaks stand in for a wine glass"}
BENCH_WATER = {"step_water": None, "drip_puddle": None, "rain_tin_roof": "water drops stand in for rain on a tin roof"}
BENCH_NO_JUDGE = {"pistol_reload": "gun mechanism references too thin on the held-out split",
                  "rifle_reload": "gun mechanism references too thin on the held-out split",
                  "ball_grass": "no reference class for a ball bounce", "ball_metal": "no reference class for a ball bounce",
                  "ball_stone": "no reference class for a ball bounce", "ball_wood": "no reference class for a ball bounce",
                  "crate_fall": "no reference class for a crate drop", "barrel_fall": "no reference class for a barrel drop",
                  "campfire": "crackling fire is a negatives-only set", "lightsaber_swing": "no reference class",
                  "grapple_fire": "no reference class", "coin_pickup": "shell-drop references too thin",
                  "monster_growl": "no reference class", "bow_release": "no reference class",
                  "laser_zap": "no reference class", "magic_missile": "no reference class", "ice_crack": "no reference class"}
ARM_TEXT = {"phrasebook": "the shipped phrasebook (--prompts tuned, the default): sarge.prompts.build, with the "
                          "--world physics template on the six physics scenes",
            "caption": "the universal template --prompts template:caption+tt: 'A single {intensity} {material} "
                       "{event} sound, recorded up close.' with 'TrackType: SFX, ' on SA3 and the contact words of "
                       "--world on physics scenes",
            "llm": "the prompt writer (--writer llama, Gemma 4 E2B-it QAT on the CPU): three fields from the key, "
                   "composed in the SA3 or the Woosh form; a rejected answer keeps the phrasebook",
            "llm_nosound": "diagnostic, no flag: the writer's source and action without its sound field, same tails",
            "llm_named": "diagnostic, no flag: the writer's prompt with the request's material and event words put "
                         "in front whenever the writer's source line lacks one of them",
            "wr_named": "--writer llama --writer-style named: v1 instructions and examples plus the naming rule (the "
                        "source keeps the request's event and material words, the validator puts them in front when "
                        "missing); hollow and static kept",
            "wr_texture": "--writer-style texture: the naming rule, a sound field on the character of the sound with no "
                          "generic collision words, four new examples (none an impact, none from an eval set); v1 form",
            "wr_title": "--writer-style title: texture's instructions and examples; Woosh gets source, sound and the "
                        "tails without the action, SA3 the v1 form",
            "wr_caption": "--writer-style caption: texture's rules with the action as one plain caption carrying the "
                          "event's words; Woosh gets action and sound, SA3 'TrackType: SFX, {Source}. {Action}, "
                          "{sound}, ...'",
            "rg_anchor": "--writer llama --writer-style named --writer-ground anchors: the named writer with the four "
                         "nearest real recording captions (caption index, CPU embedder) added to its input as style "
                         "anchors; judged against the scene facts alone",
            "rg_words": "--writer ground --writer-ground words: no language model; the request's event and material "
                        "words, then up to four sound words shared by the eight nearest real recording captions",
            "rg_caption": "--writer ground --writer-ground caption: no language model; the request's event and "
                          "material words in front of the nearest real recording caption, v1 composition",
            "ceiling": "diagnostic, no flag: one prompt per scenario written by a strong model from the scene facts "
                       "alone, in each audio model's own preferred form (the ceiling of what a writer can do)",
            "fl_phrasebook": "the phrasebook prompt rendered on Woosh Flow (non-distilled, the adopted fast path) "
                             "in place of DFlow; SA3 rows are the phrasebook clips",
            "fl_caption": "the caption template prompt rendered on Woosh Flow in place of DFlow; SA3 rows are the "
                          "caption clips",
            "fc_rk4_8": "the caption template on Woosh Flow with a fixed rk4 solver, 8 steps; SA3 rows are the caption "
                        "clips",
            "fc_rk4_16": "the caption template on Woosh Flow with a fixed rk4 solver, 16 steps; SA3 rows are the "
                         "caption clips",
            "fc_mid_16": "the caption template on Woosh Flow with a fixed midpoint solver, 16 steps; SA3 rows are the "
                         "caption clips",
            "fc_mid_32": "the caption template on Woosh Flow with a fixed midpoint solver, 32 steps; SA3 rows are the "
                         "caption clips",
            "dc_6": "the caption template on Woosh DFlow at 6 steps; SA3 rows are the caption clips",
            "dc_8": "the caption template on Woosh DFlow at 8 steps; SA3 rows are the caption clips",
            "dp_8": "the phrasebook on Woosh DFlow at 8 steps; SA3 rows are the phrasebook clips"}
TIERS = {arm: "%s_%s" % ("abcdefghijklmnopqrstuvwxyz"[i], arm) for i, arm in enumerate(ALL_ARMS)}
SET_TEXT = {"bench": "the 40 scene-bearing bench scenarios", "generic": "the 35 plain requests of the generic run",
            "fresh": "the 30 fresh scenarios (llm_eval.FRESH, or FRESH3 with --set fresh3 or all3)",
            "dev": "development scenarios (round-2 FRESH and the template round's fresh requests)"}
GROUP_TEXT = {"all": "every judged scenario", "demo": "demo scenarios, both sets", "unseen": "unseen scenarios, both sets",
              "bench": "scene-bearing bench scenarios", "bench_demo": "bench, demo", "bench_unseen": "bench, unseen",
              "generic": "plain requests of the generic run", "generic_demo": "generic run, demo",
              "generic_unseen": "generic run, unseen", "guns": "gunshots", "footsteps": "footsteps",
              "sa3_rows": "rows SA3 medium renders", "woosh_rows": "rows Woosh renders", "water": "water judge rows",
              "all_plus_water": "every judged scenario plus the water rows", "fresh": "fresh requests",
              "fresh_unseen": "fresh requests", "known": "known scenarios, both sets",
              "known_unseen": "known unseen scenarios, both sets", "water_fresh": "water judge rows, fresh",
              "water_known": "water judge rows, known", "unseen_plus_water": "unseen scenarios plus their water rows",
              "demo_plus_water": "demo scenarios plus their water rows"}
SELECT_STYLES = tuple("wr_" + s for s in ("named", "texture", "title", "caption"))
RULE = {"guns": -0.10, "water": -0.15, "demo": -0.10, "margin": 0.02, "tie": 0.005}
GOAL = {"demo": -0.05, "guns": -0.10, "water": -0.10}


def sid_of(kind, name):
    return ("b_" if kind == "bench" else "g_") + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def stem_of(sid):
    return re.sub(r"[^a-z]", "", sid) + "-s"


def fresh_rows(fresh=FRESH, label="fresh"):
    out = []
    for f in fresh:
        sc = scenes.quantise(f["scene"], None, f["event_type"]) if f.get("scene") else ""
        k = CacheKey(f["event_type"], f["material"], f.get("intensity", "normal"), f.get("context", "default"),
                     scene=sc)
        sid = "f_" + re.sub(r"[^a-z0-9]+", "_", f["id"].lower()).strip("_")
        out.append({"sid": sid, "set": label, "group": "unseen", "name": f["id"], "key": wr.key_fields(k),
                    "type": f["type"], "approx": f.get("approx"), "no_judge": None,
                    "water": stem_of(sid) if f.get("water") else None, "water_approx": None,
                    "note": f.get("note", ""), "scene_in": f.get("scene")})
    return out


def dev_rows():
    out = fresh_rows(FRESH, "dev")
    for et, mat, _g in ge.KEY_SETS["fresh"]:
        k = CacheKey(et, mat, ge.INTENSITY, ge.CONTEXT)
        kk = ge.kid(et, mat)
        sid = "t_" + re.sub(r"[^a-z0-9]+", "_", kk.lower()).strip("_")
        out.append({"sid": sid, "set": "dev", "group": "unseen", "name": kk, "key": wr.key_fields(k),
                    "type": ge.VOCAB.get(kk), "approx": ge.APPROX.get(kk), "no_judge": None, "water": None,
                    "water_approx": None, "note": "", "scene_in": None})
    return out


def build_scenarios(which="known"):
    if which not in SETS:
        raise SystemExit("scenario set %r: %s" % (which, ", ".join(SETS)))
    out, bench = [], []
    known = which in ("known", "all", "all3")
    if known:
        with open(SCENES, encoding="utf-8") as fh:
            bench = json.load(fh)
    for r in bench:
        phys = r.get("phys")
        b = bucket(phys, r["event_type"]) if isinstance(phys, dict) else ""
        sc = scenes.quantise(r["scene"], phys if isinstance(phys, dict) else None, r["event_type"])
        k = CacheKey(r["event_type"], r["material"], r["intensity"], r["context"], phys=b, scene=sc)
        sid = sid_of("bench", r["id"])
        out.append({"sid": sid, "set": "bench", "group": "demo" if r["group"] == "demo" else "unseen",
                    "name": r["id"], "key": wr.key_fields(k), "type": BENCH_TYPE.get(r["id"]),
                    "approx": BENCH_APPROX.get(r["id"]), "no_judge": BENCH_NO_JUDGE.get(r["id"]),
                    "water": stem_of(sid) if r["id"] in BENCH_WATER else None,
                    "water_approx": BENCH_WATER.get(r["id"]), "note": r.get("note", "")})
    for et, mat, g in ge.KEY_SETS["old"] if known else ():
        k = CacheKey(et, mat, ge.INTENSITY, ge.CONTEXT)
        kk = ge.kid(et, mat)
        sid = sid_of("generic", kk)
        out.append({"sid": sid, "set": "generic", "group": g, "name": kk, "key": wr.key_fields(k),
                    "type": ge.VOCAB.get(kk), "approx": ge.APPROX.get(kk), "no_judge": ge.NO_JUDGE.get(kk),
                    "water": stem_of(sid) if kk in ge.WATER else None,
                    "water_approx": ge.APPROX.get(kk) if kk in ge.WATER else None, "note": ""})
    if which in ("fresh", "all"):
        out += fresh_rows()
    if which in ("fresh3", "all3"):
        out += fresh_rows(FRESH3)
    if which == "dev":
        out += dev_rows()
    for r in out:
        k = wr.key_of(r["key"])
        r["model"] = "sa3" if side_for(k, ROUTE) == "guns" else "woosh"
        r["text"] = wr.key_text(k)
    sids = [r["sid"] for r in out]
    stems = [r["water"] for r in out if r["water"]]
    if len(set(sids)) != len(sids) or len(set(stems)) != len(stems):
        raise SystemExit("scenario ids or water stems collide")
    if not all(re.fullmatch(r"[a-z]+-[a-z]+", s) for s in stems):
        raise SystemExit("water stems must be letters only")
    return out


def render_key(r):
    k = r["key"]
    return CacheKey(k["event_type"], k["material"], k["intensity"], k["context"], k["phys"])


def phrase_prompt(k, model):
    if model == "sa3":
        return build(k.event_type, k.material, k.intensity, k.context)
    return compose("tuned", k, "")


def caption_prompt(k, model):
    return template(CAPTION, k.event_type, k.material, k.intensity, k.context, k.phys, model)


def cmp_pairs(arms):
    out = [(x, "phrasebook") for x in arms if x != "phrasebook"]
    out += [(x, "caption") for x in arms if x not in ("phrasebook", "caption")]
    out += [(x, "llm") for x in arms if (x in VARIANTS or x in STYLE_ARMS) and "llm" in arms]
    out += [(x, "wr_named") for x in arms if x in GROUND_ARMS and "wr_named" in arms]
    out += [(x, "rg_caption") for x in arms if x in EXTRA_ARMS and "rg_caption" in arms]
    return tuple(out)


def _near(w, words):
    return any(w == x or (len(w) >= 5 and w[:5] == x[:5]) for x in words)


def variant_prompt(arm, slots, k, model):
    src, act, snd = (wr._clean(slots.get(x)) for x in wr.SLOTS)
    contact, room = wr.tail_words(k)
    head = ""
    if arm == "llm_nosound":
        snd = ""
    elif arm == "llm_named":
        mf = template_slots(k.event_type, k.material, k.intensity, k.context, k.phys)["mf"]
        have = wr._words(src)
        head = mf if any(not _near(w, have) for w in wr._words(mf)) else ""
    if model == "sa3":
        first = head or src
        parts = (first[:1].upper() + first[1:], src if head else "", act, snd, contact, wr.SA3_FRAME, room)
        return wr.SA3_PREFIX + ", ".join(p for p in parts if p) + "."
    return ", ".join(p for p in (head, src, act, snd, contact, room) if p)


def cmd_variants(a):
    S = load(scen_path(a.work))
    for r in S["rows"]:
        k = wr.key_of(r["key"])
        for arm in VARIANTS:
            r["prompts"][arm] = (variant_prompt(arm, r["writer"]["slots"], k, r["model"]) if r["writer"]["ok"]
                                 else r["prompts"]["phrasebook"])
    dump(S, scen_path(a.work))
    for r in S["rows"]:
        print("%-30s %s" % (r["sid"], " || ".join(r["prompts"][x] for x in ("llm",) + VARIANTS)))
    return 0


def clip_id(model, rk, text):
    return hashlib.sha1(("%s\n%s\n%s\n%s" % (model, rk, text, AMP)).encode("utf-8")).hexdigest()[:16]


def dump(obj, p):
    ge.dump(obj, p)


def load(p):
    return ge.load_json(p)


def scen_path(work):
    return os.path.join(work, "scenarios.json")


def write_style(rows, style, a, ground=None):
    kind = "ground" if ground in ("words", "caption") else "llama"
    tag_ = (style if ground is None else (style + "_" if kind == "llama" else "") + ground)
    pw = wr.build(kind, model=a.model, port=a.port, threads=a.threads, store=os.path.join(a.work, "writer"),
                  log_path=os.path.join(a.work, "writer_server%s.log" % (
                      "" if tag_ == wr.DEFAULT_STYLE else "_" + tag_)), style=style, ground=ground,
                  ground_index=getattr(a, "ground_index", None))
    pw.setup()
    t0 = time.perf_counter()
    todo = [r for r in rows if str(wr.key_of(r["key"])) not in pw.store]
    load_s = None
    if todo:
        pw.backend.start()
        load_s = time.perf_counter() - t0
        print("writer %s style %s ready in %.1f s, affinity %s, id %s" % (
            os.path.basename(pw.backend.model), tag_, load_s, hex(pw.backend.seen_affinity or 0), pw.id[:12]),
            flush=True)
    for r in todo:
        k = wr.key_of(r["key"])
        rec = pw.write(k)
        pw.store.put(str(k), rec)
        print("%-8s %-30s %6.0f ms %-5s %s" % (tag_, r["sid"], rec["ms"], "FAIL" if rec.get("failed") else
                                              ("fix" if rec.get("repaired") else "ok"),
                                              rec.get(r["model"]) or rec.get("after") or rec.get("flags")), flush=True)
    if todo:
        pw.store.flush()
        pw.backend.close()
    return pw, load_s, [r["sid"] for r in todo]


def writer_info(rec):
    rec = rec or {}
    return {"ok": wr.ok_record(rec), "repaired": bool(rec.get("repaired")), "flags": rec.get("flags"),
            "after": rec.get("after"), "slots": rec.get("slots"), "orig": rec.get("orig"), "raw": rec.get("raw"),
            "ms": rec.get("ms"), "woosh": rec.get("woosh"), "sa3": rec.get("sa3")}


def cmd_write(a):
    rows = build_scenarios(a.set)
    os.makedirs(a.work, exist_ok=True)
    styles, arm_of = {}, {}
    for style in a.styles:
        styles[style] = write_style(rows, style, a)
        arm_of[style] = STYLE_ARM[style]
    for arm in getattr(a, "grounds", None) or ():
        mode = GROUND[arm]
        styles[arm] = write_style(rows, GROUND_BASE.get(mode, wr.DEFAULT_STYLE), a, mode)
        arm_of[arm] = arm
    for r in rows:
        k = wr.key_of(r["key"])
        r["prompts"] = {"phrasebook": phrase_prompt(k, r["model"]), "caption": caption_prompt(k, r["model"])}
        r["writers"] = {}
        for style, (pw, _l, _n) in styles.items():
            arm = arm_of[style]
            rec = pw.store.get(str(k))
            info = writer_info(rec)
            if rec and rec.get("ground"):
                info["ground"] = rec["ground"]
            r["prompts"][arm] = rec[r["model"]] if info["ok"] else phrase_prompt(k, r["model"])
            r["writers"][arm] = dict(info, style=pw.spec["name"] if style in GROUND else style, writer_id=pw.id)
            if style == wr.DEFAULT_STYLE:
                r["writer"] = {x: info[x] for x in info if x != "after"}
    smeta = {}
    for style, (pw, load_s, now) in styles.items():
        arm = arm_of[style]
        ms = sorted(r["writers"][arm]["ms"] for r in rows if r["writers"][arm]["ms"] is not None)
        ms_now = sorted(r["writers"][arm]["ms"] for r in rows if r["sid"] in set(now))
        smeta[style] = {"arm": arm, "writer_id": pw.id, "store": pw.store.path, "load_s": load_s,
                        "written_now": len(now),
                        "ms_p50": float(np.median(ms)) if ms else None,
                        "ms_p95": float(np.percentile(ms, 95)) if ms else None,
                        "ms_now_p50": float(np.median(ms_now)) if ms_now else None,
                        "ms_now_p95": float(np.percentile(ms_now, 95)) if ms_now else None,
                        "failed": [r["sid"] for r in rows if not r["writers"][arm]["ok"]],
                        "repaired": [r["sid"] for r in rows if r["writers"][arm]["repaired"]]}
    first = styles[a.styles[0]][0]
    lead = smeta[a.styles[0]]
    meta = {"writer_id": first.id, "store": first.store.path, "model": getattr(first.backend, "model", None),
            "load_s": lead["load_s"], "written_now": lead["written_now"], "route": ROUTE, "amp": AMP,
            "ms_p50": lead["ms_p50"], "failed": lead["failed"], "repaired": lead["repaired"], "set": a.set,
            "styles": smeta}
    dump({"meta": meta, "rows": rows}, scen_path(a.work))
    for style, m in smeta.items():
        print("style %-8s id %s: %d scenarios, %d written now, p50 %.0f ms, %d rejected, %d repaired" % (
            style, m["writer_id"][:12], len(rows), m["written_now"], m["ms_p50"] or 0, len(m["failed"]),
            len(m["repaired"])), flush=True)
    print("saved", scen_path(a.work), flush=True)
    return 0


def scenarios(work, only=None):
    rows = load(scen_path(work))["rows"]
    if only:
        rows = [r for r in rows if r["sid"] in only]
    return rows


def engine_of(r, arm):
    if r["model"] != "woosh":
        return r["model"]
    if arm in FLOW_BASE:
        return ENGINES["flow"]
    if arm in SPEED_BASE:
        return ENGINES[SPEED_BASE[arm][1]]
    return r["model"]


def jobs(rows, model):
    out = {}
    for r in rows:
        rk = render_key(r)
        for arm in ARMS:
            if engine_of(r, arm) != model:
                continue
            text = r["prompts"][arm]
            cid = clip_id(model, str(rk), text)
            j = out.setdefault(cid, {"cid": cid, "model": model, "rkey": rk, "text": text, "uses": []})
            j["uses"].append([r["sid"], arm])
    return [out[c] for c in sorted(out, key=lambda c: (str(out[c]["rkey"]), out[c]["text"]))]


def npy_path(work, cid, v):
    return os.path.join(work, "clips", cid, "%02d.npy" % v)


def cut_file(work, cid, v):
    return os.path.join(work, "cuts", cid, "%02d.wav" % v)


def save(work, cid, v, y, sr, first, takes):
    y = np.ascontiguousarray(y, dtype=np.float32)
    if v - first < takes:
        p = npy_path(work, cid, v)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.save(p, y)
    AudioClip(ge.judge_cut(y, sr), ge.JSR).to_wav(cut_file(work, cid, v))
    return dict(ge.facts(y, sr), sha1=hashlib.sha1(y.tobytes()).hexdigest())


def seeds(a):
    return range(a.first_seed, a.first_seed + a.seeds)


def cmd_render_woosh(a):
    sys.path.insert(1, SCRIPTS)
    import woosh_server as ws
    kind = getattr(a, "engine", "dflow")
    model = ENGINES[kind]
    base, steps, solver = ENGINE_SPEC[kind]
    opts = dict(ws.OPTIONS, **ws.ADOPTED[base])
    frames = ws.parse_frames(None)
    t0 = time.perf_counter()
    gen = ws.make_gen(base, frames["other"], steps, solver, **opts)
    ws.prepare(gen, frames)
    print("woosh %s ready in %.1f s, %s" % (kind, time.perf_counter() - t0, opts), flush=True)
    out = []
    for j in jobs(scenarios(a.work, a.only), model):
        ms = []
        for v in seeds(a):
            t = time.perf_counter()
            raw = ws.render_key(gen, j["rkey"], v, frames, "tuned", prompt=j["text"])
            ms.append((time.perf_counter() - t) * 1000.0)
            y = ws.finish_key(raw, j["rkey"], AMP, True)
            out.append(dict(save(a.work, j["cid"], v, y, ws.SR, a.first_seed, a.takes), cid=j["cid"], model=model,
                            rkey=str(j["rkey"]), prompt=j["text"], v=v, sr=ws.SR, gen_ms=round(ms[-1], 2),
                            uses=j["uses"]))
        print("woosh %-60s median %.1f ms  %s  <- %s" % (str(j["rkey"])[:60], float(np.median(ms)), j["text"],
                                                         ",".join("%s:%s" % tuple(u) for u in j["uses"])), flush=True)
    dump({"meta": {"model": "woosh-" + kind, "options": opts, "steps": steps, "solver": solver, "frames": frames,
                   "first_seed": a.first_seed,
                   "seeds": a.seeds, "takes": a.takes, "graph_error": gen.stats.get("graph_error"),
                   "nfe_median": float(np.median(gen.stats["nfe"])) if gen.stats.get("nfe") else None,
                   "seconds": time.perf_counter() - t0}, "rows": out},
         os.path.join(a.work, "render_woosh%s%s.json" % ("" if kind == "dflow" else "_" + kind, tag(a))))
    print("saved %d rows" % len(out), flush=True)
    return 0


def cmd_render_sa3(a):
    sys.path.insert(1, SCRIPTS)
    import run_sidecar as rs
    from sarge.generators.routed import WARM_KEY
    from sarge.generators.stable_audio3 import StableAudio3Generator
    kw = dict(rs.TIERS["default"], fast_load=True, stereo=True, prompt_style="detailed", amp=True)
    kw.update(rs.HYBRID_GUNS)
    wc = rs.cache_dir_of(argparse.Namespace(weight_cache=None))
    if wc:
        kw["weight_cache"] = wc
    t0 = time.perf_counter()
    g = StableAudio3Generator(**kw)
    g.ensure_ready()
    g.generate(WARM_KEY, variant=0)
    g.generate(WARM_KEY, variant=0)
    g.amp, g.amp_mode = True, AMP
    print("sa3 medium ready in %.1f s" % (time.perf_counter() - t0), flush=True)
    out = []
    for j in jobs(scenarios(a.work, a.only), "sa3"):
        ms, retries = [], 0
        for v in seeds(a):
            r0 = int(g.static_stats.get("static_retries", 0))
            t = time.perf_counter()
            clip = g.generate(j["rkey"], variant=v, prompt=j["text"])
            ms.append((time.perf_counter() - t) * 1000.0)
            rr = int(g.static_stats.get("static_retries", 0)) - r0
            retries += rr
            out.append(dict(save(a.work, j["cid"], v, clip.samples, clip.sample_rate, a.first_seed, a.takes),
                            cid=j["cid"], model="sa3", rkey=str(j["rkey"]), prompt=j["text"], v=v,
                            sr=int(clip.sample_rate), gen_ms=round(ms[-1], 2), static_retries=rr, uses=j["uses"]))
        print("sa3   %-40s median %.1f ms  retries %d  %s  <- %s" % (
            str(j["rkey"])[:40], float(np.median(ms)), retries, j["text"],
            ",".join("%s:%s" % tuple(u) for u in j["uses"])), flush=True)
    dump({"meta": {"model": "sa3-medium", "kwargs": kw, "first_seed": a.first_seed, "seeds": a.seeds, "takes": a.takes,
                   "fast_stats": g.fast_stats(), "seconds": time.perf_counter() - t0}, "rows": out},
         os.path.join(a.work, "render_sa3%s.json" % tag(a)))
    print("saved %d rows" % len(out), flush=True)
    return 0


def tag(a):
    return "_" + a.tag if getattr(a, "tag", "") else ""


def render_rows(work):
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(work, "render_*.json"))):
        rows += load(p)["rows"]
    return rows


def arm_clip(r, arm):
    return clip_id(engine_of(r, arm), str(render_key(r)), r["prompts"][arm])


def judged(rows):
    return [r for r in rows if r["type"]]


def cmd_routes(a):
    rows = scenarios(a.work)
    for arm in ARMS:
        d = os.path.join(a.work, "routes", arm)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        for r in judged(rows):
            cid = arm_clip(r, arm)
            for v in seeds(a):
                ge.link(cut_file(a.work, cid, v), os.path.join(d, "%s-s_%02d.wav" % (r["sid"], v - a.first_seed)))
        dump({"clusters": [], "sources": [], "steps": None,
              "prompts": {"%s|s" % r["sid"]: r["prompts"][arm] for r in judged(rows)},
              "note": "%s; served path rendered in process (woosh_server.render_key + finish_key with --amp full and "
                      "stereo, StableAudio3Generator with run_sidecar HYBRID_GUNS), route %s; judged clip = mono 0.6 s "
                      "_extract window at normalize_loudness target_for('normal'), 44.1 kHz; variants %d-%d, numbered "
                      "from 00" % (ARM_TEXT[arm], ROUTE, a.first_seed, a.first_seed + a.seeds - 1)},
             os.path.join(d, "route.json"))
    wd = a.water_dir
    if os.path.isdir(wd):
        shutil.rmtree(wd)
    for arm in ARMS:
        d = os.path.join(wd, TIERS[arm])
        os.makedirs(d)
        for r in rows:
            if not r["water"]:
                continue
            cid = arm_clip(r, arm)
            for v in seeds(a):
                ge.link(cut_file(a.work, cid, v), os.path.join(d, "%s_%02d.wav" % (r["water"], v - a.first_seed)))
    os.makedirs(os.path.join(a.work, "water_empty"), exist_ok=True)
    print("routes: %d judged scenarios per arm, water tiers %s" % (len(judged(rows)), sorted(os.listdir(wd))), flush=True)
    return 0


def cmd_judge_vocab(a):
    import vocab_eval as ve
    ve.KEYS = [(r["sid"], "s", r["type"]) for r in judged(scenarios(a.work))]
    argv = ["--refs", a.refs, "--settings", "none", "--seeds", str(a.seeds), "--out",
            os.path.join(a.work, "vocab%s.json" % tag(a)), "--listen", os.path.join(a.work, "vocab_listen%s.wav" % tag(a))]
    for arm in ARMS:
        argv += ["--route", "%s=%s" % (arm, os.path.join(a.work, "routes", arm))]
    return ve.main(argv)


def cmd_judge_water(a):
    import water_judge as wj
    return wj.main(["--ctl-lengths", "0.6", "--probe-json", os.path.join(a.work, "no_probe.json"), "--probe-dir",
                    os.path.join(a.work, "water_empty"), "--tier-dir", a.water_dir, "--out",
                    os.path.join(a.work, "water%s.json" % tag(a))])


def vocab_rows(V, rows):
    cons, gen = V["consensus"], V["generated"]
    shape = V.get("shape", {})
    ident = gen.get("_identity", {})
    out = {}
    for r in judged(rows):
        arms = {}
        for arm in ARMS:
            g = "gen:%s|%s|s" % (arm, r["sid"])
            c = cons.get(g)
            if not c or not c.get("judges"):
                break
            arms[arm] = {"cons": c.get("cons"), "cons_ci": c.get("cons_ci"),
                         "judges": {f: {"variant": j["variant"], "rate": j["rate"], "real": j["real"]}
                                    for f, j in c["judges"].items()},
                         "hits": {f: gen[g][j["variant"]]["hits"] for f, j in c["judges"].items()},
                         "shape": shape.get(g), "identity": ident.get(g)}
        if len(arms) != len(ARMS):
            continue
        base = arms["phrasebook"]
        out[r["sid"]] = {"sid": r["sid"], "set": r["set"], "group": r["group"], "type": r["type"], "approx": r["approx"],
                         "model": r["model"], "reals": {f: j["real"] for f, j in base["judges"].items()},
                         "variants": {f: j["variant"] for f, j in base["judges"].items()},
                         "arms": {arm: arms[arm]["hits"] for arm in ARMS},
                         "cons_ci": {arm: arms[arm]["cons_ci"] for arm in ARMS},
                         "shape": {arm: arms[arm]["shape"] for arm in ARMS},
                         "identity": {arm: arms[arm]["identity"] for arm in ARMS}}
    return out


def water_rows(W, rows):
    ch = W["selection"]["chosen"].get("0.6") or {}
    val = W["validity"]
    gen = W["generated"]
    out = {}
    for r in rows:
        if not r["water"]:
            continue
        wk = "%s|%s" % tuple(r["water"].split("-"))
        arms, reals, variants = {}, {}, {}
        for arm in ARMS:
            tg = "gen:tier|" + TIERS[arm]
            hits = {}
            for f, v in ch.items():
                rec = gen.get(v, {}).get(tg)
                if not rec:
                    continue
                hs = sorted((int(p[0].split("#")[1]), int(p[1])) for p in rec["pairs"] if p[0].split("#")[0] == wk)
                hits[f] = [x for _s, x in hs]
                reals[f] = val[v]["0.6"]["real"]
                variants[f] = v
            if hits:
                arms[arm] = hits
        if len(arms) != len(ARMS):
            continue
        out[r["sid"]] = {"sid": r["sid"], "set": r["set"], "group": "water", "kind": r["group"], "type": "water",
                         "approx": r["water_approx"], "model": r["model"], "reals": reals, "variants": variants,
                         "arms": arms, "cons_ci": {}}
    rng = np.random.default_rng(ge.BOOT_SEED + 1)
    for k, w in out.items():
        for arm in ARMS:
            n = len(next(iter(w["arms"][arm].values())))
            idx = rng.integers(0, n, size=(ge.BOOT, n))
            cb = ge.cons_draws(w["arms"][arm], w["reals"], idx)
            w["cons_ci"][arm] = [float(x) for x in np.percentile(cb, [2.5, 97.5])]
    return out


def groups_of(J):
    ks = list(J)
    g = {"all": ks, "demo": [k for k in ks if J[k]["group"] == "demo"],
         "unseen": [k for k in ks if J[k]["group"] == "unseen"]}
    for s in ("bench", "generic", "fresh"):
        g[s] = [k for k in ks if J[k]["set"] == s]
        for grp in ("demo", "unseen"):
            g["%s_%s" % (s, grp)] = [k for k in ks if J[k]["set"] == s and J[k]["group"] == grp]
    if g["fresh"] and len(g["fresh"]) < len(ks):
        g["known_unseen"] = [k for k in g["unseen"] if J[k]["set"] != "fresh"]
        g["known"] = [k for k in ks if J[k]["set"] != "fresh"]
    g["guns"] = [k for k in ks if J[k]["type"].startswith("gunshot")]
    g["footsteps"] = [k for k in ks if J[k]["type"] == "footstep"]
    g["sa3_rows"] = [k for k in ks if J[k]["model"] == "sa3"]
    g["woosh_rows"] = [k for k in ks if J[k]["model"] == "woosh"]
    return {n: m for n, m in g.items() if m}


def fmt_ci(v, ci, sign=True):
    return ge.fmt_ci(v, ci, sign)


def cons(J, k, arm):
    return ge.cons_of(J[k]["arms"][arm], J[k]["reals"])


def clip_facts(rows, rr):
    by = {}
    for x in rr:
        by.setdefault(x["cid"], []).append(x)
    out = {}
    for r in rows:
        out[r["sid"]] = {}
        for arm in ARMS:
            xs = by.get(arm_clip(r, arm), [])
            out[r["sid"]][arm] = {"cid": arm_clip(r, arm), "n": len(xs),
                                  "t40": float(np.median([x["t40"] for x in xs])) if xs else None,
                                  "len": float(np.median([x["len"] for x in xs])) if xs else None,
                                  "static": int(sum(x["static"] for x in xs)),
                                  "gen_ms": float(np.median([x["gen_ms"] for x in xs])) if xs else None}
    return out


def fmt_shape(sh):
    return ge.fmt_shape(sh)


def fmt_ident(x):
    return ge.fmt_ident(x)


def water_merged(work):
    import glob
    ps = sorted(glob.glob(os.path.join(work, "water*.json")))
    if not ps:
        return None
    W = load(ps[0])
    for p in ps[1:]:
        X = load(p)
        if X["selection"]["chosen"].get("0.6") != W["selection"]["chosen"].get("0.6"):
            raise SystemExit("%s chose other water judges" % p)
        for v, by in X["generated"].items():
            W["generated"].setdefault(v, {}).update(by)
    return W


def cmd_report(a):
    CMP = cmp_pairs(ARMS)
    ge.RUN, ge.CMP = ARMS, CMP
    rng = np.random.default_rng(ge.BOOT_SEED)
    S = load(scen_path(a.work))
    rows = S["rows"]
    by_sid = {r["sid"]: r for r in rows}
    V = load(os.path.join(a.work, "vocab.json"))
    ge.merge_vocab(V, a.work)
    J = vocab_rows(V, rows)
    WM = water_merged(a.work)
    W = water_rows(WM, rows) if WM else {}
    facts = clip_facts(rows, render_rows(a.work))
    per = {}
    for k, r in J.items():
        per[k] = {"set": r["set"], "group": r["group"], "type": r["type"], "approx": r["approx"], "model": r["model"],
                  "cons": {arm: cons(J, k, arm) for arm in ARMS}, "cons_ci": r["cons_ci"],
                  "cmp": {"%s vs %s" % (x, y): ge.paired_key(r, x, y, rng) for x, y in CMP},
                  "identity": r["identity"], "shape": r["shape"], "judges": r["variants"]}
    water = {}
    for k, r in W.items():
        water[k] = {"set": r["set"], "kind": r["kind"], "approx": r["approx"], "model": r["model"],
                    "cons": {arm: cons(W, k, arm) for arm in ARMS}, "cons_ci": r["cons_ci"],
                    "cmp": {"%s vs %s" % (x, y): ge.paired_key(r, x, y, rng) for x, y in CMP}, "judges": r["variants"]}
    G = {n: ge.group_stats(m, J, rng) for n, m in groups_of(J).items()}
    if W:
        G["water"] = ge.group_stats(list(W), W, rng)
        for s in ("bench", "generic", "fresh"):
            m = [k for k in W if W[k]["set"] == s]
            if m:
                G["water_" + s] = ge.group_stats(m, W, rng)
        AW = dict(J, **{"water:" + k: r for k, r in W.items()})
        G["all_plus_water"] = ge.group_stats(list(AW), AW, rng)
        UW = dict({k: J[k] for k in J if J[k]["group"] == "unseen"},
                  **{"water:" + k: r for k, r in W.items() if r["kind"] == "unseen"})
        DW = dict({k: J[k] for k in J if J[k]["group"] == "demo"},
                  **{"water:" + k: r for k, r in W.items() if r["kind"] == "demo"})
        for g, m in (("unseen_plus_water", UW), ("demo_plus_water", DW)):
            if m:
                G[g] = ge.group_stats(list(m), m, rng)
    res = {"meta": dict(S["meta"], first_seed=a.first_seed, seeds=a.seeds, boot=ge.BOOT, refs=V["meta"].get("manifest")),
           "per": per, "water": water, "groups": G, "clips": facts,
           "prompts": {r["sid"]: {"set": r["set"], "group": r["group"], "name": r["name"], "model": r["model"],
                                  "text": r["text"], "prompts": r["prompts"], "writer": r["writer"],
                                  "type": r["type"], "no_judge": r["no_judge"]} for r in rows},
           "validity": V.get("chosen")}
    dump(res, os.path.join(a.work, "report%s.json" % tag(a)))
    lines = []
    say = lines.append
    say("LLM PROMPT WRITER ON SOUND: %d scenarios (%s), variants %d-%d, route %s, --amp %s, stereo" % (
        len(rows), "; ".join(SET_TEXT[s] for s in SET_TEXT if any(r["set"] == s for r in rows)), a.first_seed,
        a.first_seed + a.seeds - 1, ROUTE, AMP))
    say("writer %s (id %s), rejected %s, repaired %s" % (os.path.basename(str(S["meta"].get("model"))),
                                                         str(S["meta"].get("writer_id"))[:12],
                                                         S["meta"]["failed"] or "none", S["meta"]["repaired"] or "none"))
    for style, m in (S["meta"].get("styles") or {}).items():
        if m["arm"] in ARMS:
            say("style %s as arm %s: writer id %s, %d rejected, %d repaired of the written scenarios" % (
                style, m["arm"], m["writer_id"][:12], len(m["failed"]), len(m["repaired"])))
    say("cons = min over valid held-out judges of min(rate / real, 1); deltas paired on seed, 95%% CI from %d draws; "
        "McNemar a/p/c = AST/PANNs/CLAP seeds gained/lost" % ge.BOOT)
    say("")
    say("PER SCENARIO (identity judges)")
    say("%-34s %-16s %-6s %-6s %s" % ("scenario", "type", "group", "model", " ".join("%-18s" % x for x in ARMS)))
    for k, r in per.items():
        say("%-34s %-16s %-6s %-6s %s" % (k, r["type"] + ("*" if r["approx"] else ""), r["group"], r["model"], " ".join(
            "%-18s" % fmt_ci(r["cons"][arm], r["cons_ci"][arm], False) for arm in ARMS)))
        for x, y in CMP:
            c = r["cmp"]["%s vs %s" % (x, y)]
            say("%36s %-24s %-26s p %.3f  %s" % ("", "%s vs %s" % (x, y), fmt_ci(c["delta"], c["ci"]), c["p_boot"],
                                                ge.fmt_mc(c["mcnemar"])))
    say("")
    say("WATER JUDGE")
    for k, r in water.items():
        say("%-34s %-6s %s" % (k + ("*" if r["approx"] else ""), r["kind"], " ".join(
            "%-18s" % fmt_ci(r["cons"][arm], r["cons_ci"][arm], False) for arm in ARMS)))
        for x, y in CMP:
            c = r["cmp"]["%s vs %s" % (x, y)]
            say("%36s %-24s %-26s p %.3f  %s" % ("", "%s vs %s" % (x, y), fmt_ci(c["delta"], c["ci"]), c["p_boot"],
                                                ge.fmt_mc(c["mcnemar"])))
    say("")
    say("POOLED (mean of cons over scenarios [95% CI, scenarios and seeds resampled])")
    for g, s in G.items():
        say(" %s, %d rows: %s" % (GROUP_TEXT.get(g, g), len(s["keys"]), "  ".join(
            "%s %s" % (arm, fmt_ci(s["arms"][arm]["mean"], s["arms"][arm]["ci"], False)) for arm in ARMS)))
        for name, c in s["cmp"].items():
            say("    %-24s %-26s p %.3f  rows up/down %d/%d (sign p %.3f)  judges %s" % (
                name, fmt_ci(c["delta"], c["ci"]), c["p_boot"], c["up"], c["down"], c["sign_p"], " ".join(
                    "%s +%d/-%d p%.3f" % (f, x["gain"], x["loss"], x["p"]) for f, x in c["judges"].items())))
    for arm in ARMS:
        others = [o for o in ARMS if o != arm]
        hit = [k for k, r in per.items() if r["cons"][arm] < 0.5 or any(
            r["cons"][o] - r["cons"][arm] >= 0.2 for o in others)]
        if not hit:
            continue
        say("")
        say("WHERE %s FAILS (cons under 0.5, or 0.2 or more under another arm): CLAP top-1, shape medians, prompt" % arm)
        for k in hit:
            r, cl = per[k], facts[k]
            say(" %s %s (%s): %s" % (k, r["type"], r["model"], "  ".join("%s %.2f" % (o, r["cons"][o]) for o in ARMS)))
            for o in ARMS:
                say("    %-10s CLAP %s" % (o, fmt_ident(r["identity"][o])))
                say("    %-10s %s; -40 dB after %.2f s, length %.2f s" % ("", fmt_shape(r["shape"][o]), cl[o]["t40"] or 0,
                                                                         cl[o]["len"] or 0))
                say("    %-10s prompt: %s" % ("", by_sid[k]["prompts"][o]))
    wfail = [k for k, r in water.items() if any(r["cons"][o] < 0.5 for o in ARMS)]
    if wfail:
        say("")
        say("WATER ROWS WITH AN ARM UNDER 0.5")
        for k in wfail:
            say(" %s: %s" % (k, "  ".join("%s %.2f" % (o, water[k]["cons"][o]) for o in ARMS)))
            for o in ARMS:
                say("    %-10s prompt: %s" % (o, by_sid[k]["prompts"][o]))
    say("")
    say("UNJUDGED SCENARIOS (page only): prompts and clip facts")
    for r in rows:
        if r["type"] and r["sid"] in per:
            continue
        if r["water"] and r["sid"] in water:
            continue
        say(" %s (%s, %s): %s" % (r["sid"], r["model"], r["group"], r["no_judge"] or "no judge"))
        for o in ARMS:
            f = facts[r["sid"]][o]
            say("    %-10s -40 dB after %.2f s, length %.2f s, static %d | %s" % (
                o, f["t40"] or 0, f["len"] or 0, f["static"], r["prompts"][o]))
    with open(os.path.join(a.work, "report%s.txt" % tag(a)), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved", os.path.join(a.work, "report%s.json" % tag(a)))
    return 0


def joined(parts):
    J, W = {}, {}
    for work in parts:
        rows = load(scen_path(work))["rows"]
        V = load(os.path.join(work, "vocab.json"))
        ge.merge_vocab(V, work)
        WM = water_merged(work)
        for out, part in ((J, vocab_rows(V, rows)), (W, water_rows(WM, rows) if WM else {})):
            for k, r in part.items():
                if k not in out:
                    out[k] = dict(r, arms={a: {f: list(h) for f, h in r["arms"][a].items()} for a in ARMS})
                    continue
                if out[k]["variants"] != r["variants"]:
                    raise SystemExit("%s: judges differ between %s and the first part" % (k, work))
                for a in ARMS:
                    for f, h in r["arms"][a].items():
                        out[k]["arms"][a][f] += list(h)
    return J, W


def cmd_combine(a):
    CMP = cmp_pairs(ARMS)
    ge.RUN, ge.CMP = ARMS, CMP
    rng = np.random.default_rng(ge.BOOT_SEED)
    J, W = joined(a.parts)
    G = {n: ge.group_stats(m, J, rng) for n, m in groups_of(J).items()}
    if W:
        G["water"] = ge.group_stats(list(W), W, rng)
        AW = dict(J, **{"water:" + k: r for k, r in W.items()})
        G["all_plus_water"] = ge.group_stats(list(AW), AW, rng)
    per = {k: {"cons": {arm: cons(J, k, arm) for arm in ARMS},
               "cmp": {"%s vs %s" % (x, y): ge.paired_key(J[k], x, y, rng) for x, y in CMP}} for k in J}
    wper = {k: {"cons": {arm: cons(W, k, arm) for arm in ARMS},
                "cmp": {"%s vs %s" % (x, y): ge.paired_key(W[k], x, y, rng) for x, y in CMP}} for k in W}
    n = len(next(iter(next(iter(J.values()))["arms"]["phrasebook"].values())))
    dump({"parts": a.parts, "seeds_per_row": n, "groups": G, "per": per, "water": wper}, a.out)
    print("COMBINED over %s: %d seeds per row" % (", ".join(a.parts), n))
    for g, s in G.items():
        print(" %s, %d rows: %s" % (GROUP_TEXT.get(g, g), len(s["keys"]), "  ".join(
            "%s %s" % (arm, fmt_ci(s["arms"][arm]["mean"], s["arms"][arm]["ci"], False)) for arm in ARMS)))
        for name, c in s["cmp"].items():
            print("    %-26s %-26s p %.3f  rows up/down %d/%d  judges %s" % (
                name, fmt_ci(c["delta"], c["ci"]), c["p_boot"], c["up"], c["down"], " ".join(
                    "%s +%d/-%d" % (f[0], x["gain"], x["loss"]) for f, x in c["judges"].items())))
    print("saved", a.out)
    return 0


def cmd_subset(a):
    S = load(scen_path(a.source))
    keep = [r for r in S["rows"] if (r["set"] == "fresh") == (a.set == "fresh")]
    if not keep:
        raise SystemExit("no %s rows in %s" % (a.set, scen_path(a.source)))
    p = scen_path(a.work)
    new = {"meta": dict(S["meta"], subset=a.set, source=scen_path(a.source)), "rows": keep}
    if os.path.isfile(p) and load(p) != json.loads(json.dumps(new)):
        raise SystemExit("%s holds other scenarios; pick a new work folder" % p)
    os.makedirs(a.work, exist_ok=True)
    dump(new, p)
    print("subset %s: %d of %d rows to %s" % (a.set, len(keep), len(S["rows"]), p), flush=True)
    return 0


def rule_numbers(G, arm):
    d = {n: G[g]["cmp"]["%s vs phrasebook" % arm]["delta"]
         for n, g in (("U", "unseen"), ("D", "demo"), ("G", "guns"), ("W", "water"), ("A", "all_plus_water"))}
    d["Q"] = (d["U"] + d["D"]) / 2.0
    return d


def pick_style(num, rejected, styles=SELECT_STYLES):
    elig = [s for s in styles if num[s]["G"] >= RULE["guns"] and num[s]["W"] >= RULE["water"]
            and num[s]["D"] >= RULE["demo"]]
    if not elig:
        return None, elig, []
    best = max(num[s]["Q"] for s in elig)
    tied = [s for s in elig if num[s]["Q"] >= best - RULE["tie"]]
    pick = sorted(tied, key=lambda s: (-num[s]["A"], rejected.get(s, 0), styles.index(s)))[0]
    return pick, elig, tied


def decide_selection(num, rejected, styles=SELECT_STYLES, ref="llm"):
    pick, elig, tied = pick_style(num, rejected, styles)
    go = pick is not None and num[pick]["Q"] >= num[ref]["Q"] + RULE["margin"]
    out = {"pick": pick, "eligible": elig, "tied": tied, "confirm": go}
    if ref != "llm":
        out["ref"] = ref
    return out


def cmd_select(a):
    rep = load(os.path.join(a.work, "report%s.json" % tag(a)))
    G = rep["groups"]
    ref = getattr(a, "ref", "llm")
    styles = [s for s in getattr(a, "candidates", SELECT_STYLES) if s in ARMS and s != ref]
    meta = {m["arm"]: m for m in (rep["meta"].get("styles") or {}).values()}
    rejected = {s: len(meta[s]["failed"]) for s in styles if s in meta}
    num = {arm: rule_numbers(G, arm) for arm in [ref] + styles}
    dec = decide_selection(num, rejected, tuple(styles), ref)
    n = {k: len(G[g]["keys"]) for k, g in (("U", "unseen"), ("D", "demo"), ("G", "guns"), ("W", "water"),
                                            ("A", "all_plus_water"))}
    lines = ["SELECTION, block S: variants %d-%d, %s; computed by llm_eval.py select from %s" % (
        a.first_seed, a.first_seed + a.seeds - 1, ", ".join(sorted({r["set"] for r in rep["prompts"].values()})),
        os.path.join(a.work, "report%s.json" % tag(a))),
        "rows: U %(U)d unseen identity, D %(D)d demo identity, G %(G)d guns, W %(W)d water, A %(A)d" % n,
        "every number is the arm minus the phrasebook, mean over rows, paired on seed", "",
        "%-11s %-13s %8s %8s %8s %8s %8s %8s  %s" % ("arm", "writer id", "U", "D", "G", "W", "A", "Q", "eligible")]
    stay = "the writer stays at v1 and off" if ref == "llm" else "every default stays"
    for arm in [ref] + styles:
        x = num[arm]
        wid = (meta.get(arm) or {}).get("writer_id", "")[:12]
        el = "reference" if arm == ref else ("yes" if arm in dec["eligible"] else "no")
        lines.append("%-11s %-13s %+8.4f %+8.4f %+8.4f %+8.4f %+8.4f %+8.4f  %s" % (
            arm, wid, x["U"], x["D"], x["G"], x["W"], x["A"], x["Q"], el))
    lines.append("")
    lines.append("bars: G >= %.2f, W >= %.2f, D >= %.2f; margin Q_pick >= Q_%s + %.2f; tie within %.3f goes to "
                 "higher A, then fewer rejected prompts, then the order %s" % (
                     RULE["guns"], RULE["water"], RULE["demo"], ref, RULE["margin"], RULE["tie"], ", ".join(styles)))
    lines.append("rejected prompts over the written scenarios: %s" % ", ".join(
        "%s %d" % (s, rejected.get(s, 0)) for s in styles))
    lines.append("styles within the tie band of the best eligible Q: %s" % (", ".join(dec["tied"]) or "none"))
    if dec["pick"] is None:
        lines.append("RESULT: no style is eligible; nothing is rendered for confirmation, %s" % stay)
    else:
        lines.append("best eligible style: %s, Q %+.4f against Q_%s %+.4f, margin %+.4f" % (
            dec["pick"], num[dec["pick"]]["Q"], ref, num[ref]["Q"], num[dec["pick"]]["Q"] - num[ref]["Q"]))
        lines.append("RESULT: PICK %s, goes on to confirmation" % dec["pick"] if dec["confirm"] else
                     "RESULT: %s misses the margin over %s; nothing is rendered for confirmation, %s" % (
                         dec["pick"], "v1" if ref == "llm" else ref, stay))
    text = "\n".join(lines) + "\n"
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    dump(dict(dec, numbers=num, rows=n, rejected=rejected, report=os.path.join(a.work, "report%s.json" % tag(a))),
         os.path.splitext(a.out)[0] + ".json")
    print(text)
    print("saved", a.out)
    return 0


def part_first(work):
    import glob
    firsts = {load(p)["meta"]["first_seed"] for p in glob.glob(os.path.join(work, "render_*.json"))}
    if len(firsts) != 1:
        raise SystemExit("%s: render files disagree on the first seed (%s)" % (work, sorted(firsts)))
    return firsts.pop()


def goal_verdict(G, pick):
    pv = "%s vs phrasebook" % pick
    u, d, g, w = (G[n]["cmp"][pv] for n in ("unseen", "demo", "guns", "water"))
    out = {"C1 unseen": {"delta": u["delta"], "ci": u["ci"], "rows": len(G["unseen"]["keys"]),
                         "pass": u["delta"] > 0 and u["ci"][0] > 0, "bar": "delta > 0 and its 95% lower bound > 0"},
           "C2 demo": {"delta": d["delta"], "ci": d["ci"], "rows": len(G["demo"]["keys"]),
                       "pass": d["delta"] >= GOAL["demo"], "bar": "delta >= %.2f" % GOAL["demo"]},
           "C3 guns": {"delta": g["delta"], "ci": g["ci"], "rows": len(G["guns"]["keys"]),
                       "pass": g["delta"] >= GOAL["guns"], "bar": "delta >= %.2f" % GOAL["guns"]},
           "C4 water": {"delta": w["delta"], "ci": w["ci"], "rows": len(G["water"]["keys"]),
                        "pass": w["delta"] >= GOAL["water"], "bar": "delta >= %.2f" % GOAL["water"]}}
    return out, all(c["pass"] for c in out.values())


def fmt_judges(c):
    return " ".join("%s +%d/-%d p%.3f" % (f, x["gain"], x["loss"], x["p"]) for f, x in c["judges"].items())


CONFIRM_GROUPS = ("unseen", "demo", "guns", "water", "fresh", "known_unseen", "water_fresh", "water_known", "all",
                  "all_plus_water", "unseen_plus_water", "footsteps", "sa3_rows", "woosh_rows")


def cmd_confirm(a):
    pick = a.pick
    if pick not in ARMS or "phrasebook" not in ARMS or "caption" not in ARMS:
        raise SystemExit("--arms must hold phrasebook, caption and the pick %s" % pick)
    CMP = cmp_pairs(ARMS)
    ge.RUN, ge.CMP = ARMS, CMP
    rng = np.random.default_rng(ge.BOOT_SEED)
    J, W = joined(a.parts)
    rows, facts, span = {}, {}, {}
    for work in a.parts:
        rr = scenarios(work)
        f0 = part_first(work)
        n = {len(next(iter(J[r["sid"]]["arms"]["phrasebook"].values()))) for r in rr if r["sid"] in J}
        span[work] = {"first_seed": f0, "seeds": sorted(n), "rows": len(rr), "sets": sorted({r["set"] for r in rr})}
        for r in rr:
            if r["sid"] in rows:
                raise SystemExit("%s is in two parts" % r["sid"])
            rows[r["sid"]] = r
        facts.update(clip_facts(rr, render_rows(work)))
    G = {n: ge.group_stats(m, J, rng) for n, m in groups_of(J).items()}
    if W:
        G["water"] = ge.group_stats(list(W), W, rng)
        for s, keep in (("fresh", lambda r: r["set"] == "fresh"), ("known", lambda r: r["set"] != "fresh")):
            m = [k for k in W if keep(W[k])]
            if m and len(m) < len(W):
                G["water_" + s] = ge.group_stats(m, W, rng)
        AW = dict(J, **{"water:" + k: r for k, r in W.items()})
        G["all_plus_water"] = ge.group_stats(list(AW), AW, rng)
        UW = dict({k: J[k] for k in J if J[k]["group"] == "unseen"},
                  **{"water:" + k: r for k, r in W.items() if r["kind"] == "unseen"})
        G["unseen_plus_water"] = ge.group_stats(list(UW), UW, rng)
    verdict, meets = goal_verdict(G, pick)
    per = {k: {"set": r["set"], "group": r["group"], "type": r["type"], "approx": r["approx"], "model": r["model"],
               "cons": {arm: cons(J, k, arm) for arm in ARMS}, "cons_ci": r["cons_ci"],
               "cmp": {"%s vs %s" % (x, y): ge.paired_key(r, x, y, rng) for x, y in CMP},
               "identity": r["identity"], "shape": r["shape"], "judges": r["variants"]} for k, r in J.items()}
    water = {k: {"set": r["set"], "kind": r["kind"], "approx": r["approx"], "model": r["model"],
                 "cons": {arm: cons(W, k, arm) for arm in ARMS}, "cons_ci": r["cons_ci"],
                 "cmp": {"%s vs %s" % (x, y): ge.paired_key(r, x, y, rng) for x, y in CMP}, "judges": r["variants"]}
             for k, r in W.items()}
    res = {"pick": pick, "arms": list(ARMS), "parts": span, "goal": verdict, "meets_goal": meets, "groups": G,
           "per": per, "water": water, "clips": facts,
           "prompts": {k: {"set": r["set"], "group": r["group"], "name": r["name"], "model": r["model"],
                           "text": r["text"], "prompts": r["prompts"], "writers": r.get("writers")}
                       for k, r in rows.items()}}
    lines = []
    say = lines.append
    say("CONFIRMATION of %s against the phrasebook and the caption template, paired on seed" % pick)
    for work, s in span.items():
        say(" part %s: %s, %d scenarios, variants %d-%d" % (work, "+".join(s["sets"]), s["rows"], s["first_seed"],
                                                            s["first_seed"] + max(s["seeds"]) - 1))
    say("arms %s; llm is v1, context only; cons = min over valid held-out judges of min(rate / real, 1); 95%% CI from "
        "%d draws, rows and seeds resampled; judges a/p/c = AST/PANNs/CLAP seeds gained/lost with the McNemar sign p"
        % (", ".join(ARMS), ge.BOOT))
    say("")
    say("GOAL VERDICT (%s minus the phrasebook)" % pick)
    for name, c in verdict.items():
        say(" %-10s %2d rows  %-26s  bar %-40s %s" % (name, c["rows"], fmt_ci(c["delta"], c["ci"]), c["bar"],
                                                      "PASS" if c["pass"] else "FAIL"))
    say(" RESULT: %s" % ("the pick meets the goal" if meets else "the pick does not meet the goal"))
    say("")
    say("POOLED (mean of cons over rows [95% CI])")
    for g in [x for x in CONFIRM_GROUPS if x in G] + [x for x in G if x not in CONFIRM_GROUPS]:
        s = G[g]
        say(" %s, %d rows: %s" % (GROUP_TEXT.get(g, g), len(s["keys"]), "  ".join(
            "%s %s" % (arm, fmt_ci(s["arms"][arm]["mean"], s["arms"][arm]["ci"], False)) for arm in ARMS)))
        for name, c in s["cmp"].items():
            say("    %-26s %-26s p %.3f  rows up/down %d/%d (sign p %.3f)  judges %s" % (
                name, fmt_ci(c["delta"], c["ci"]), c["p_boot"], c["up"], c["down"], c["sign_p"], fmt_judges(c)))
    for title, table in (("PER SCENARIO (identity judges)", per), ("WATER JUDGE", water)):
        say("")
        say(title)
        say("%-30s %-16s %-6s %-6s %-6s %s" % ("scenario", "type", "set", "group", "model",
                                               " ".join("%-18s" % x for x in ARMS)))
        for k, r in sorted(table.items(), key=lambda kv: (kv[1]["set"], kv[0])):
            say("%-30s %-16s %-6s %-6s %-6s %s" % (k, (r.get("type") or "water") + ("*" if r["approx"] else ""),
                                                   r["set"], r.get("group", r.get("kind")), r["model"], " ".join(
                "%-18s" % fmt_ci(r["cons"][arm], r["cons_ci"][arm], False) for arm in ARMS)))
            for x, y in CMP:
                c = r["cmp"]["%s vs %s" % (x, y)]
                say("%32s %-26s %-26s p %.3f  %s" % ("", "%s vs %s" % (x, y), fmt_ci(c["delta"], c["ci"]), c["p_boot"],
                                                    ge.fmt_mc(c["mcnemar"])))
    lose = [k for k, r in per.items() if r["cons"][pick] < r["cons"]["phrasebook"] - 1e-9 or
            r["cons"][pick] < r["cons"]["caption"] - 1e-9]
    say("")
    say("WHERE %s LOSES (under the phrasebook or the caption template): CLAP top-1, shape medians, prompts" % pick)
    for k in sorted(lose, key=lambda k: per[k]["cons"][pick] - max(per[k]["cons"]["phrasebook"],
                                                                     per[k]["cons"]["caption"])):
        r, cl = per[k], facts[k]
        say(" %s %s (%s, %s): %s" % (k, r["type"], r["set"], r["model"], "  ".join(
            "%s %.2f" % (o, r["cons"][o]) for o in ARMS)))
        for o in ARMS:
            say("    %-11s CLAP %s" % (o, fmt_ident(r["identity"][o])))
            say("    %-11s %s; -40 dB after %.2f s, length %.2f s" % ("", fmt_shape(r["shape"][o]), cl[o]["t40"] or 0,
                                                                      cl[o]["len"] or 0))
            say("    %-11s prompt: %s" % ("", rows[k]["prompts"][o]))
    wl = [k for k, r in water.items() if r["cons"][pick] < max(r["cons"]["phrasebook"], r["cons"]["caption"]) - 1e-9]
    if wl:
        say("")
        say("WATER ROWS WHERE %s LOSES" % pick)
        for k in wl:
            say(" %s: %s" % (k, "  ".join("%s %.2f" % (o, water[k]["cons"][o]) for o in ARMS)))
            for o in ARMS:
                say("    %-11s prompt: %s" % (o, rows[k]["prompts"][o]))
    text = "\n".join(lines) + "\n"
    dump(res, a.out)
    with open(os.path.splitext(a.out)[0] + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print("saved", a.out)
    return 0


PAGE_TEXT = {"title": "Prompt Writer Listening", "id": "llm", "intro": """<h1>Prompt writer: blind listening</h1>
<p>Use headphones. Each row is one moment a game sends, rendered on the models that ship: SA3 medium for whatever the
route calls a gun, Woosh DFlow for everything else. The options are three prompt sources for the same request on the
same seeds: the shipped phrasebook, the universal caption template, and the prompt the small CPU language model wrote
from the scenario. Two sources that wrote the same text share one option, so a row has two or three options. Every
option plays %TAKES% takes (variants 40 to 43, which no earlier run rendered). Letters are shuffled per row.</p>
<p>The first two sections are the 40 bench scenarios, which carry the scene record (who or what made the sound, the
object, what it touched, the room); the facts under each title are what the writer read. The last two are the 35 plain
requests of the generic run, where the writer saw only the event and the material. Levels are what the game plays.
Mark the option that sounds most like the moment and the one that sounds least like it, or press "none sound right".
Picks stay in this browser. Reveal names the prompt source behind each option, the model, the prompt and the judge
score where a validated judge exists; Download saves your picks with the key.</p>"""}
PAGE2_TEXT = {"title": "Prompt Writer Round Two", "id": "llm2", "intro": """<h1>Prompt writer, round two: blind listening</h1>
<p>Use headphones. Each row is one moment a game sends, rendered on the models that ship: SA3 medium for whatever the
route calls a gun, Woosh DFlow for everything else. The options are three prompt sources for the same request on the
same seeds: the shipped phrasebook, the universal caption template, and the prompt the small CPU language model wrote
from the scenario under its new general rules. Two sources that wrote the same text share one option, so a row has two
or three options. Every option plays %TAKES% takes. On the 75 known scenarios they are variants 100 to 103; the last
section holds 30 fresh requests that no run had rendered before, at variants 0 to 3. Letters are shuffled per row.</p>
<p>The bench sections and half of the fresh requests carry the scene record (who or what made the sound, the object,
what it touched, the room); the facts under each title are what the writer read. The other rows are plain requests,
where the writer saw only the event and the material. Levels are what the game plays. Mark the option that sounds most
like the moment and the one that sounds least like it, or press "none sound right". Picks stay in this browser. Reveal
names the prompt source behind each option, the model, the prompt and the judge score where a validated judge exists;
Download saves your picks with the key.</p>"""}
PAGE3_TEXT = {"title": "Grounded Prompt Listening", "id": "rag", "intro": """<h1>Caption-grounded prompts: blind listening</h1>
<p>Use headphones. Each row is one moment a game sends, rendered on the models that ship: SA3 medium for whatever the
route calls a gun, Woosh DFlow for everything else. The options are six prompt sources for the same request on the
same seeds: the shipped phrasebook, the universal caption template, the small CPU writer in its named style, and three
prompts grounded in the captions of real recordings nearest to the request (the writer reading those captions, the
words they share, and the nearest caption itself). Two sources that wrote the same text share one option, so a row has
up to six options. Every option plays %TAKES% takes, variants 120 to 123 of the selection block; no confirmation block
was rendered, because the preregistered rule stopped after selection. Letters are shuffled per row.</p>
<p>The bench sections carry the scene record (who or what made the sound, the object, what it touched, the room); the
facts under each title are what the prompt sources read. The other rows are plain requests, with only the event and
the material. Levels are what the game plays. Mark the option that sounds most like the moment and the one that sounds
least like it, or press "none sound right". Picks stay in this browser. Reveal names the prompt source behind each
option, the model, the prompt and the judge score where a validated judge exists; Download saves your picks with the
key.</p>"""}
PAGES = {"llm": PAGE_TEXT, "llm2": PAGE2_TEXT, "rag": PAGE3_TEXT}
PAGE_SECTIONS = (("bench", "demo", "Demo moments with the scene record"), ("bench", "unseen", "Other moments with the scene record"),
                 ("generic", "demo", "Demo requests, plain keys"), ("generic", "unseen", "Unseen requests, plain keys"),
                 ("fresh", "unseen", "Fresh requests, never rendered before"))


def row_title(r):
    k = r["key"]
    head = "%s | %s" % (k["event_type"], k["material"]) if k["material"] else k["event_type"]
    if r["set"] == "generic" or not r.get("scene_in", r["set"] == "bench"):
        return head, ""
    facts = "; ".join(ln for ln in r["text"].split("\n")[1:])
    return head, facts


def page_rows(a):
    out = []
    for work in a.parts or [a.work]:
        meta = {(x["cid"], x["v"]): x for x in render_rows(work)}
        first = part_first(work) if a.parts else a.first_seed
        out += [(r, work, meta, first) for r in scenarios(work)]
    return out


def cmd_page(a):
    rep = load(a.report or os.path.join(a.work, "report%s.json" % tag(a)))
    out = a.out
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "a"))
    key, info, n_rows, n_clips = {}, {}, 0, 0
    sec = {(s, g): [] for s, g, _t in PAGE_SECTIONS}
    for r, work, meta, first in page_rows(a):
        sid = r["sid"]
        opts = {}
        for arm in ARMS:
            opts.setdefault(arm_clip(r, arm), []).append(arm)
        order = sorted(opts, key=lambda c: ge.h(a.seed + sid + c))
        blocks = []
        for i, cid in enumerate(order):
            arms = opts[cid]
            clips, nums = [], []
            for v in range(first, first + a.takes):
                y = np.load(npy_path(work, cid, v))
                x = meta[(cid, v)]
                nm = ge.h(a.seed + sid + cid + str(v))[:12] + ".wav"
                AudioClip(np.ascontiguousarray(y, dtype=np.float32), int(x["sr"])).to_wav(os.path.join(out, "a", nm))
                n_clips += 1
                nums.append(x)
                clips.append("<audio controls preload='none' src='a/%s' aria-label='%s take %d'></audio>" % (
                    nm, ge.LETTERS[i], v - first + 1))
            letter = ge.LETTERS[i]
            key["%s|%s" % (sid, letter)] = arms
            judged_ = []
            pk = rep["per"].get(sid)
            if pk:
                judged_.append("identity %.2f [%.2f, %.2f]" % (pk["cons"][arms[0]], pk["cons_ci"][arms[0]][0],
                                                              pk["cons_ci"][arms[0]][1]))
            wk = rep["water"].get(sid)
            if wk:
                judged_.append("water %.2f [%.2f, %.2f]" % (wk["cons"][arms[0]], wk["cons_ci"][arms[0]][0],
                                                           wk["cons_ci"][arms[0]][1]))
            info["%s|%s" % (sid, letter)] = "%s | %s, amp %s | %s | takes (variants %d-%d): -40 dB after %s s, length %s s | prompt: %s" % (
                ", ".join(arms), MODEL_NAME[engine_of(r, arms[0])], AMP, "; ".join(judged_) or "no validated judge for this sound",
                first, first + a.takes - 1, " / ".join("%.2f" % n["t40"] for n in nums),
                " / ".join("%.2f" % n["len"] for n in nums), nums[0]["prompt"])
            blocks.append("<div class='opt' data-s='%s' data-l='%s'><div class='let'>%s</div><div class='takes'>%s</div>"
                          "<div class='btns'><button data-v='best'>best</button><button data-v='worst'>worst</button>"
                          "</div><div class='rev'></div></div>" % (sid, letter, letter, "".join(clips)))
        head, facts = row_title(r)
        sub = "<p class='facts'>%s</p>" % html.escape(facts) if facts else ""
        sec[(r["set"], r["group"])].append(
            "<section class='snd' data-s='%s'><h3>%s</h3>%s%s<p class='row'><button class='none' data-s='%s'>none sound "
            "right</button><span class='st'></span></p></section>" % (sid, html.escape(head), sub, "".join(blocks), sid))
        n_rows += 1
    body = "".join("<h2>%s</h2>%s" % (t, "".join(sec[(s, g)])) for s, g, t in PAGE_SECTIONS if sec[(s, g)])
    arm_text = {x: ARM_TEXT[x] for x in ARMS} if a.parts else ARM_TEXT
    payload = base64.b64encode(json.dumps({"key": key, "info": info, "arms": arm_text}, ensure_ascii=True).encode()).decode()
    page = ge.fill_page(PAGES[a.intro], body, payload, n_rows, "sarge_llm_page_" + a.seed, a.takes)
    page = page.replace("</style>", ".facts{font-size:13px;margin:0 0 6px;overflow-wrap:anywhere}</style>", 1)
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    dump({"seed": a.seed, "key": key, "info": info}, os.path.join(os.path.dirname(os.path.abspath(out)), "page_key.json"))
    print("wrote %s (%d rows, %d clips)" % (os.path.join(out, "index.html"), n_rows, n_clips), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="the prompt writer against the phrasebook and the caption template on the "
                                             "models that ship: write, render, judge, report, blind page")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("write", "variants", "render-woosh", "render-sa3", "routes", "judge-vocab", "judge-water", "report",
                 "page", "combine", "subset", "select", "confirm"):
        p = sub.add_parser(name)
        p.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(BASE_ARMS))
        p.add_argument("--tag", default="")
        p.add_argument("--work", default=WORK)
        p.add_argument("--first-seed", type=int, default=FIRST)
        p.add_argument("--seeds", type=int, default=SEEDS)
        p.add_argument("--takes", type=int, default=TAKES)
        p.add_argument("--water-dir", default=WATER_DIR)
        if name == "write":
            p.add_argument("--model", default=wr.DEFAULT_MODEL)
            p.add_argument("--port", type=int, default=11546)
            p.add_argument("--threads", type=int, default=4)
            p.add_argument("--styles", nargs="+", choices=list(wr.STYLES), default=[wr.DEFAULT_STYLE],
                           help="writer styles to write, each in a separate server session and store")
            p.add_argument("--grounds", nargs="*", choices=list(GROUND_ARMS), default=[],
                           help="caption-grounded arms to write: rg_anchor (named style with retrieved captions, "
                                "a server session), rg_words and rg_caption (no model)")
            p.add_argument("--ground-index", default=None, help="caption index folder (default sarge.ground)")
            p.add_argument("--set", choices=list(SETS), default="known",
                           help="known: the 40 bench scenes and 35 generic requests; fresh: llm_eval.FRESH; all: both; "
                                "fresh3: llm_eval.FRESH3; all3: known and FRESH3; dev: FRESH and the template round's "
                                "fresh requests, for development")
        if name in ("render-woosh", "render-sa3"):
            p.add_argument("--only", nargs="+", default=None)
        if name == "render-woosh":
            p.add_argument("--engine", choices=sorted(ENGINES), default="dflow",
                           help="dflow renders every arm on the served DFlow; flow renders only the Flow arms "
                                "(fl_phrasebook, fl_caption) on Woosh Flow with woosh_server.ADOPTED['flow']; the "
                                "other engines (ENGINE_SPEC: fixed-step Flow solvers, DFlow at 6 or 8 steps) render "
                                "only the arms of SPEED_BASE that name them")
        if name == "judge-vocab":
            p.add_argument("--refs", default=os.path.join("data", "refs_heldout"))
        if name == "page":
            p.add_argument("--out", default=PAGE)
            p.add_argument("--seed", default="llm1")
            p.add_argument("--parts", nargs="+", default=None,
                           help="work folders whose rows all go on one page, each part at the first seed it was rendered from")
            p.add_argument("--report", default=None, help="report json with per and water by scenario")
            p.add_argument("--intro", choices=sorted(PAGES), default="llm")
        if name == "subset":
            p.add_argument("--source", required=True, help="work folder whose scenarios.json is filtered")
            p.add_argument("--set", choices=["known", "fresh"], required=True)
        if name == "select":
            p.add_argument("--out", required=True, help="selection.txt; the same name with .json gets the numbers")
            p.add_argument("--ref", default="llm", choices=list(ALL_ARMS),
                           help="the arm the pick must beat by the margin (llm in round 2, wr_named in round 3)")
            p.add_argument("--candidates", nargs="+", default=list(SELECT_STYLES),
                           choices=list(STYLE_ARMS + GROUND_ARMS), help="arms to pick from, in tie-break order")
        if name == "confirm":
            p.add_argument("--parts", nargs="+", required=True, help="work folders with disjoint rows")
            p.add_argument("--pick", required=True, choices=list(STYLE_ARMS + GROUND_ARMS + EXTRA_ARMS + FLOW_ARMS))
            p.add_argument("--out", required=True)
        if name == "combine":
            p.add_argument("--parts", nargs="+", required=True, help="work folders whose judged seeds are pooled per row")
            p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    global ARMS
    ARMS = tuple(a.arms)
    a.water_dir = os.path.abspath(os.path.join(ROOT, a.water_dir))
    return {"write": cmd_write, "variants": cmd_variants, "render-woosh": cmd_render_woosh, "render-sa3": cmd_render_sa3, "routes": cmd_routes,
            "judge-vocab": cmd_judge_vocab, "judge-water": cmd_judge_water, "report": cmd_report,
            "page": cmd_page, "combine": cmd_combine, "subset": cmd_subset, "select": cmd_select,
            "confirm": cmd_confirm}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
