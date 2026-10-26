"""Easy testing and training screen.   Run:  streamlit run app.py
1) Measure tab: upload photos, see measurements and the debug pictures, type the tailor's tape values, save.
2) Training tab: see saved customers and press one button to train the correction."""
import csv
import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from calibrate import CALIB_PATH, MIN_ROWS, NAMES, train
from measure import measure

DATA = Path(__file__).with_name("data")
CSV = DATA / "training.csv"
LABEL = {"shoulder": "Shoulder width", "sleeve": "Sleeve length", "inseam": "Inseam",
         "outseam": "Outseam (waist to floor)", "chest": "Chest", "waist": "Waist", "hip": "Hips"}
COLS = ["id", "date", "height_cm", "used_side_photo"] + [f"{p}_{n}" for n in NAMES for p in ("raw", "tape")]

st.set_page_config(page_title="Measurement lab", layout="wide")
st.title("Measurement lab")
tab1, tab2 = st.tabs(["1. Measure and save", "2. Training"])

with tab1:
    c1, c2, c3 = st.columns(3)
    height = c1.number_input("Customer height (cm)", 120.0, 220.0, 170.0, 0.5)
    front = c2.file_uploader("Front photo", ["jpg", "jpeg", "png"])
    side = c3.file_uploader("Side photo (recommended)", ["jpg", "jpeg", "png"])
    if st.button("Measure", type="primary", disabled=front is None):
        with st.spinner("Measuring..."):
            try:
                st.session_state.res = measure(front, height, side, debug=True)
                st.session_state.photos = (front.getvalue(), side.getvalue() if side else None)
                st.session_state.height = height
            except ValueError as e:
                st.session_state.res = None
                st.error(str(e))
    res = st.session_state.get("res")
    if res:
        for w in res["warnings"]:
            st.warning(w)
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Result")
            st.dataframe(pd.DataFrame({"Measurement": [LABEL[n] for n in NAMES],
                                       "Result (cm)": [res["measurements_cm"][n] for n in NAMES],
                                       "Raw model (cm)": [res["raw_cm"][n] for n in NAMES]}), hide_index=True)
            if not CALIB_PATH.exists():
                st.caption("No correction trained yet, so results are the raw model output.")
        with right:
            st.subheader("What the model saw")
            st.caption("Pink = body found, yellow lines = where it measured, blue dots = joints. "
                       "If a yellow line is in the wrong place, the measurement will be wrong.")
            ov = res["overlays"]
            pics = [(ov["front"], "Front")] + ([(ov["side"], "Side")] if ov["side"] else [])
            for col, (im, cap) in zip(st.columns(len(pics)), pics):
                col.image(im, caption=cap)

        st.subheader("Save for training")
        with st.form("save"):
            cust = st.text_input("Customer name or ID")
            st.caption("Type what your tailor measures with a tape (cm). Leave 0 for any you did not measure.")
            cols = st.columns(len(NAMES))
            tape = {n: c.number_input(LABEL[n], 0.0, 300.0, 0.0, 0.5) for n, c in zip(NAMES, cols)}
            keep = st.checkbox("The customer agreed that we may also keep their photos")
            if st.form_submit_button("Save this customer"):
                if not cust.strip() or not any(tape.values()):
                    st.error("Enter a customer name or ID and at least one tape measurement.")
                else:
                    DATA.mkdir(exist_ok=True)
                    new = not CSV.exists()
                    row = {"id": cust.strip(), "date": datetime.date.today().isoformat(),
                           "height_cm": st.session_state.height, "used_side_photo": res["used_side_photo"]}
                    for n in NAMES:
                        row[f"raw_{n}"] = res["raw_cm"][n] if res["raw_cm"][n] is not None else ""
                        row[f"tape_{n}"] = tape[n] if tape[n] > 0 else ""
                    with open(CSV, "a", newline="") as f:
                        wr = csv.DictWriter(f, COLS)
                        if new:
                            wr.writeheader()
                        wr.writerow(row)
                    if keep:
                        (DATA / "photos").mkdir(exist_ok=True)
                        for tag, b in zip(("front", "side"), st.session_state.photos):
                            if b:
                                (DATA / "photos" / f"{cust.strip()}_{tag}.jpg").write_bytes(b)
                    st.success(f"Saved {cust.strip()}.")

with tab2:
    df = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(columns=COLS)
    st.metric("Customers saved", len(df))
    counts = {LABEL[n]: int(df[f"tape_{n}"].notna().sum()) if f"tape_{n}" in df else 0 for n in NAMES}
    st.write(f"Each measurement needs at least {MIN_ROWS} customers with a tape value before it can be trained.")
    st.bar_chart(pd.Series(counts, name="Customers with a tape value"))
    if st.button("Train correction", type="primary", disabled=len(df) == 0):
        st.dataframe(pd.DataFrame(train(CSV)), hide_index=True)
        st.caption("Errors are measured on customers left out of the training, so they show how it will do on new people.")
    if len(df):
        st.download_button("Download training data (CSV)", df.to_csv(index=False), "training.csv")
        with st.expander("Saved customers"):
            st.dataframe(df, hide_index=True)
    if CALIB_PATH.exists() and st.button("Remove trained correction"):
        CALIB_PATH.unlink()
        st.rerun()
