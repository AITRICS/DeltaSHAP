from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings(action="ignore")

FEATURE_MAP = {
    "Height": 0,
    "Hours": 1,
    "Diastolic Blood Pressure": 2,
    "Fraction Inspired Oxygen": 3,
    "Glucose": 4,
    "Heart Rate": 5,
    "Mean Arterial Pressure": 6,
    "Oxygen Saturation": 7,
    "Respiratory Rate": 8,
    "Systolic Blood Pressure": 9,
    "Temperature": 10,
    "Weight": 11,
    "pH": 12,
    "Capillary Refill Rate": 13,
    "Glascow Coma Scale Eye Opening 0": 14,
    "Glascow Coma Scale Eye Opening 1": 15,
    "Glascow Coma Scale Eye Opening 2": 16,
    "Glascow Coma Scale Eye Opening 3": 17,
    "Glascow Coma Scale Motor Response 0": 18,
    "Glascow Coma Scale Motor Response 1": 19,
    "Glascow Coma Scale Motor Response 2": 20,
    "Glascow Coma Scale Motor Response 3": 21,
    "Glascow Coma Scale Motor Response 4": 22,
    "Glascow Coma Scale Motor Response 5": 23,
    "Glascow Coma Scale Total 0": 24,
    "Glascow Coma Scale Total 1": 25,
    "Glascow Coma Scale Total 2": 26,
    "Glascow Coma Scale Total 3": 27,
    "Glascow Coma Scale Total 4": 28,
    "Glascow Coma Scale Total 5": 29,
    "Glascow Coma Scale Total 6": 30,
    "Glascow Coma Scale Total 7": 31,
    "Glascow Coma Scale Total 8": 32,
    "Glascow Coma Scale Total 9": 33,
    "Glascow Coma Scale Total 10": 34,
    "Glascow Coma Scale Total 11": 35,
    "Glascow Coma Scale Total 12": 36,
    "Glascow Coma Scale Verbal Response 0": 37,
    "Glascow Coma Scale Verbal Response 1": 38,
    "Glascow Coma Scale Verbal Response 2": 39,
    "Glascow Coma Scale Verbal Response 3": 40,
    "Glascow Coma Scale Verbal Response 4": 41
}

# FEATURE_MAP = { # PhysioNet 2019
#     "Gender": 0,
#     "Unit1": 1,
#     "Unit2": 2,
#     "Age": 3,
#     "HospAdmTime": 4,
#     "Hours": 5,
#     "HR": 6,
#     "SpO2": 7,
#     "Temp": 8,
#     "SBP": 9,
#     "MAP": 10,
#     "DBP": 11,
#     "RR": 12,
#     "EtCO2": 13,
#     "BE": 14,
#     "HCO3": 15,
#     "FiO2": 16,
#     "pH": 17,
#     "PaCO2": 18,
#     "SaO2": 19,
#     "AST": 20,
#     "BUN": 21,
#     "AlkPhos": 22,
#     "Ca": 23,
#     "Cl": 24,
#     "Cr": 25,
#     "BiliD": 26,
#     "Glucose": 27,
#     "Lactate": 28,
#     "Mg": 29,
#     "Phos": 30,
#     "K": 31,
#     "BiliT": 32,
#     "TropI": 33,
#     "Hct": 34,
#     "Hgb": 35,
#     "PTT": 36,
#     "WBC": 37,
#     "Fib": 38,
#     "Plt": 39
# }

# FEATURE_NAMES = {
#     "Patient Gender": 0,
#     "Hospital Unit Type 1": 1,
#     "Hospital Unit Type 2": 2,
#     "Patient Age": 3,
#     "Hospital Admission Time": 4,
#     "Hours Since Admission": 5,
#     "Heart Rate": 6,
#     "Oxygen Saturation": 7,
#     "Body Temperature": 8,
#     "Systolic Blood Pressure": 9,
#     "Mean Arterial Pressure": 10,
#     "Diastolic Blood Pressure": 11,
#     "Respiratory Rate": 12,
#     "End-tidal CO2": 13,
#     "Base Excess": 14,
#     "Bicarbonate": 15,
#     "Fraction of Inspired O2": 16,
#     "Blood pH": 17,
#     "Partial Pressure CO2": 18,
#     "Arterial O2 Saturation": 19,
#     "Aspartate Aminotransferase": 20,
#     "Blood Urea Nitrogen": 21,
#     "Alkaline Phosphatase": 22,
#     "Calcium": 23,
#     "Chloride": 24,
#     "Creatinine": 25,
#     "Direct Bilirubin": 26,
#     "Blood Glucose": 27,
#     "Blood Lactate": 28,
#     "Magnesium": 29,
#     "Phosphate": 30,
#     "Potassium": 31,
#     "Total Bilirubin": 32,
#     "Troponin I": 33,
#     "Hematocrit": 34,
#     "Hemoglobin": 35,
#     "Partial Thromboplastin Time": 36,
#     "White Blood Cell Count": 37,
#     "Fibrinogen": 38,
#     "Platelet Count": 39
# }

FEATURE_RANGE = {
    "PULSE": (0, 300),
    "RESP": (0, 120),
    "SBP": (0, 300),
    "DBP": (0, 300),
    "TEMP": (25, 50),
    "SpO2": (0, 100),  # Changed from SPO2 to SpO2
    "GCS": (3, 15),
    "BILIRUBIN": (0, 75),
    "LACTATE": (0, 20),
    "CREATININE": (0, 20),
    "PLATELET": (0, 1000),
    "APH": (0, 14),  # Changed from PH to APH
    "SODIUM": (0, 500),
    "POTASSIUM": (0, 15),
    "HEMATOCRIT": (0, 100),
    "WBC": (0, 100),
    "HCO3": (0, 100),
    "CRP": (0, 900),
}

# Normalize the normal values using feature ranges
NORMAL_VALUE = {
    feature: (value - FEATURE_RANGE[feature][0])
    / (FEATURE_RANGE[feature][1] - FEATURE_RANGE[feature][0])
    for feature, value in {
        "PULSE": 80,  # Median of 60-100 bpm
        "RESP": 16,  # Median of 12-20 breaths/min
        "SBP": 105,  # Median of 90-120 mmHg
        "DBP": 70,  # Median of 60-80 mmHg
        "TEMP": 37,  # Median of 36.5-37.5°C
        "SpO2": 100.0,
        "GCS": 15,
        "BILIRUBIN": 0.6,
        "LACTATE": 0.7,
        "CREATININE": 0.8,
        "PLATELET": 300,
        "APH": 7.4,
        "SODIUM": 140,
        "POTASSIUM": 4.2,
        "HEMATOCRIT": 45,
        "WBC": 7,
        "HCO3": 24,
        "CRP": 1.5,
    }.items()
}

# Mean and standard deviation for MIMIC-III features (for unnormalization)
FEATURE_MEAN = {
    "Height": 169.205709,
    "Hours": 130.768304,
    "Capillary Refill Rate": 0.147333,
    "Diastolic Blood Pressure": 61.089489,
    "Fraction Inspired Oxygen": 0.502612,
    "Glascow Coma Scale Eye Opening": 3.441144,
    "Glascow Coma Scale Motor Response": 5.312734,
    "Glascow Coma Scale Total": 9.706133,
    "Glascow Coma Scale Verbal Response": 3.143159,
    "Glucose": 137.852357,
    "Heart Rate": 85.581663,
    "Mean Arterial Pressure": 79.419124,
    "Oxygen Saturation": 96.740761,
    "Respiratory Rate": 19.819856,
    "Systolic Blood Pressure": 122.518329,
    "Temperature": 36.942249,
    "Weight": 84.240189,
    "pH": 7.143808
}

FEATURE_STD = {
    "Height": 11.948060,
    "Hours": 196.464241,
    "Capillary Refill Rate": 0.354437,
    "Diastolic Blood Pressure": 14.652923,
    "Fraction Inspired Oxygen": 0.167981,
    "Glascow Coma Scale Eye Opening": 0.918765,
    "Glascow Coma Scale Motor Response": 1.403024,
    "Glascow Coma Scale Total": 3.745524,
    "Glascow Coma Scale Verbal Response": 1.909381,
    "Glucose": 51.831027,
    "Heart Rate": 17.527301,
    "Mean Arterial Pressure": 15.729320,
    "Oxygen Saturation": 4.820498,
    "Respiratory Rate": 6.204572,
    "Systolic Blood Pressure": 22.909507,
    "Temperature": 0.777391,
    "Weight": 24.830271,
    "pH": 0.707209
}

# Map feature indices to their names for easier lookup
FEATURE_INDEX_TO_NAME = {v: k for k, v in FEATURE_MAP.items()}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="spike", help="Dataset")
    parser.add_argument("--delay", type=int, default=0, choices=[0, 1, 2, 3, 4], help="Simulation Spike Delay amount")
    parser.add_argument("--explainer", nargs="+", type=str, default=["winit"], help="Explainer model")
    parser.add_argument("--cv", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="CV to run")
    parser.add_argument("--testbs", type=int, default=-1, help="test batch size")
    parser.add_argument("--dataseed", type=int, default=1234, help="random state for data split")
    parser.add_argument("--datapath", type=str, default=None, help="path to data")
    parser.add_argument("--explainerseed", type=int, default=2345, help="random state for explainer")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")

    # paths
    parser.add_argument("--outpath", type=str, default="./output/", help="path to output files")
    parser.add_argument("--ckptpath", type=str, default="./ckpt/", help="path to model and generator files")
    parser.add_argument("--plotpath", type=str, default="./plots/", help="path to plotting results")
    parser.add_argument("--logpath", type=str, default="./logs/", help="path to logging output")
    parser.add_argument("--resultfile", type=str, default="results.csv", help="result csv file name")
    parser.add_argument("--logfile", type=str, default=None, help="log file name")

    # run options
    parser.add_argument("--train", action="store_true", help="Run model training")
    parser.add_argument("--traingen", action="store_true", help="Run generator training")
    parser.add_argument("--skipexplain", action="store_true", help="skip explanation generation")
    parser.add_argument("--eval", action="store_true", help="run feature importance evalation")
    parser.add_argument("--drop", type=str, nargs="+", default=["local"], help="drop features")
    parser.add_argument("--cum", type=eval, default=True, help="run feature importance evalation in cumulative setting (only for int k)")
    parser.add_argument("--loglevel", type=str, default="info", choices=["warning", "info", "error", "debug"], help="Logging level")
    parser.add_argument("--nondeterministic", action="store_true", help="Non-deterministic pytorch.")

    # model args and train lr
    parser.add_argument("--hiddensize", type=int, default=200, help="hidden size for base model")
    parser.add_argument("--batchsize", type=int, default=128, help="batch size for base model")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate for base model")
    parser.add_argument("--numlayers", type=int, choices=[1, 2, 3], default=1, help="number of layers in an RNN-based base model")
    parser.add_argument("--modeltype", type=str, default="gru", help="model architecture type for base model")
    parser.add_argument("--lr", type=float, default=None, help="learning rate for model training")

    # generator args
    parser.add_argument("--joint", action="store_true", help="use joint generator")
    parser.add_argument("--conditional", action="store_true", help="use conditional generator")

    # dynamask args
    parser.add_argument("--area", type=float, nargs="+", default=None, help="Dynamask Area")
    parser.add_argument("--blurtype", type=str, choices=["gaussian", "fadema"], default=None, help="Dynamask blur type")
    parser.add_argument("--deletion", type=bool, default=None, help="Run Dynamask in deletion mode")
    parser.add_argument("--sizereg", type=int, default=None, help="Dynamask size regulator param")
    parser.add_argument("--timereg", type=int, default=None, help="Dynamask time regulator param")
    parser.add_argument("--loss", type=str, choices=["logloss", "ce"], default=None, help="Dynamask loss type")
    parser.add_argument("--epoch", type=int, default=200, help="Dynamask Epochs")
    parser.add_argument("--last_timestep_only", type=eval, default=True, help="Dynamask use last timestep only")

    # WinIT args (and FIT args for samples)
    parser.add_argument("--window", nargs="+", type=int, default=[10], help="WinIT window size")
    parser.add_argument("--winitmetric", type=str, nargs="+", choices=["kl", "js", "pd"], default=["pd"], help="WinIT metrics for divergence of distributions")
    parser.add_argument("--usedatadist", action="store_true", help="Use samples from data distribution instead of generator to generate masked features in WinIT")
    parser.add_argument("--samples", type=int, default=-1, help="Number of samples in generating masked features in WinIT")
    parser.add_argument("--top_p", type=float, default=0.2, help="Top p for masked features in WinIT")

    # DeltaSHAP args
    parser.add_argument("--deltashap_n_samples", type=int, default=25, help="Number of samples for DeltaSHAP")
    parser.add_argument("--deltashap_normalize", type=eval, default=True, help="Normalize DeltaSHAP scores")
    parser.add_argument("--deltashap_baseline", type=str, default="carryforward", choices=["carryforward", "zero"], help="Baseline for DeltaSHAP")

    # eval args
    parser.add_argument("--maskseed", type=int, default=43814, help="Seed for tie breaker on importance scores.")
    parser.add_argument("--mask", type=str, nargs="+", default=["point"], help="Mask strategy")
    parser.add_argument("--top", type=int, default=25, help="Percents to mask per time series")
    parser.add_argument("--aggregate", type=str, nargs="+", default=["mean"], choices=["mean", "max", "absmax"], help="Aggregating methods for observations over windows.")
    parser.add_argument("--numensemble", type=int, default=1, help="Number of ensemble models")

    parser.add_argument("--vis", type=eval, default=False, help="Visualize explanations")
    parser.add_argument("--vis_dir", type=str, default="plots", help="Visualize explanations")
    parser.add_argument("--num_vis", type=int, default=10, help="Number of samples to visualize")
    return parser.parse_args()
