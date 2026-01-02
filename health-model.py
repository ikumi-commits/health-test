
import os
import streamlit as st
import pickle
import numpy as np
import shap
import gdown
import pandas as pd
import google.generativeai as genai
import time
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader



st.set_page_config(layout="wide")

#タイトル
st.markdown("""
<h1 style="
    color: #2563eb;
    border-bottom: 3px solid #2563eb;
    padding-bottom: 0.3em;
    margin-bottom: 0.8em;
">
糖尿病リスクシミュレーション
</h1>
""", unsafe_allow_html=True)


# session_state 初期化
if "predicted" not in st.session_state:
    st.session_state["predicted"] = False

if "prob" not in st.session_state:
    st.session_state["prob"] = None

if "suppress_factors" not in st.session_state:
    st.session_state["suppress_factors"] = []

if "increase_factors" not in st.session_state:
    st.session_state["increase_factors"] = []

if "intro_text" not in st.session_state:
    st.session_state["intro_text"] = ""

if "target_factors" not in st.session_state:
    st.session_state["target_factors"] = []


# 同意状態の初期化
if "agreed" not in st.session_state:
    st.session_state["agreed"] = False


# 同意画面
if not st.session_state["agreed"]:
    st. write("#### ★ご利用にあたっての注意★")

    st.markdown(
        "<small>1.本サイトは、生活習慣の改善による健康増進を目的としたものであり、"
        "疾病の診断・治療・予防を目的としたものではありません。</small>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<small>2.本サイトで表示される予測値は、過去のデータ傾向をもとに予測されるものであり、"
        "医学的判断により予測をするものではありません。</small>",
        unsafe_allow_html=True
    )

    st.write("")  # 余白

    if st.button("同意する"):
        st.session_state["agreed"] = True
        with st.spinner("お待ちください..."):
            time.sleep(2)
        st.rerun()

    # 同意前はここで処理を止める
    st.stop()



# モデルの読み込み
FILE_ID = "1Mh7btoQb9QYpGg0KHhzIrpHhegG5ocq2"
MODEL_LOCAL_PATH = "rf_model.pkl"

# モデル読み込み関数
@st.cache_resource
def load_model():
    # ファイルがなければ Google Drive からダウンロード
    if not os.path.exists(MODEL_LOCAL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_LOCAL_PATH, quiet=False)

    # ファイルがあることを確認してから読み込む
    if os.path.exists(MODEL_LOCAL_PATH):
        with open(MODEL_LOCAL_PATH, "rb") as f:
            model = pickle.load(f)
        return model

    else:
        st.error("モデルファイルが存在しません。")
        st.stop()

# 実際にロード
model = load_model()

#特徴量を読み込み
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

feature_labels = {
    "HighBP": "高血圧",
    "HighChol": "高コレステロール",
    "CholCheck": "コレステロール検査済み",
    "Smoker": "喫煙習慣",
    "Stroke": "脳卒中",
    "HeartDiseaseorAttack": "心臓病・心筋梗塞",
    "PhysActivity": "運動習慣",
    "HvyAlcoholConsump": "飲酒習慣",
    "DiffWalk": "歩行や階段昇降の支障",
    "Sex": "性別",
    "Age": "年齢",
    "BMI": "BMI",
    "GenHlth": "主観的な健康状態",
    "PhysHlth": "身体の不調日数（過去30日）",
    "MentHlth": "メンタルの不調日数（過去30日）",
    "Income": "所得",
    "Fruits": "果物摂取習慣（１日に１回以上食べる）",
    "Veggies": "野菜摂取（１日に１回以上食べる）"
}

#年齢カテゴリ
age_options = {
    1: "18～24歳",
    2: "25～29歳",
    3: "30～34歳",
    4: "35～39歳",
    5: "40～44歳",
    6: "45～49歳",
    7: "50～54歳",
    8: "55～59歳",
    9: "60～64歳",
    10: "65～69歳",
    11: "70～74歳",
    12: "75～79歳",
    13: "80歳以上"
}

# 所得カテゴリ（円換算）
income_options = {
    1: "〜150万円未満",
    2: "150〜300万円未満",
    3: "300〜375万円未満",
    4: "375〜525万円未満",
    5: "525〜675万円未満",
    6: "675〜900万円未満",
    7: "900〜1125万円未満",
    8: "1125万円以上"
}

# 主観的健康状態
genhlth_options = {
            1: "非常に良い",
            2: "とても良い",
            3: "良い",
            4: "普通",
            5: "悪い"
        }


ordered_features = ["Sex", "Age", "BMI","Stroke","HeartDiseaseorAttack"] + [f for f in feature_names if f not in ["Sex", "Age", "BMI","Stroke","CholCheck","HeartDiseaseorAttack","AnyHealthcare","NoDocbcCost"]]
inputs = {}

#前提説明
st.markdown("#### 🧾 このアプリでわかること")
st.markdown(
    "- あなたと似た状態の方が糖尿病を発症した割合\n"
    "- 発症リスクと関係がある現在の生活習慣\n"
    "- 今後に向けた生活のヒント"
)

# 入力フォーム作成
def load_css():
    st.markdown("""
    <style>
    div[data-testid="stFormSubmitButton"] button {
        background-color: #27ae60;  /* 濃い緑 */
        color: white;               /* 文字を白 */
        border: none;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #1e8449;  /* ホバー時さらに濃く */
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

basic_features = [
    "Sex", "Age", "BMI", "Income"
]

history_features = [
    "HighBP",
    "HighChol",
    "Stroke",
    "HeartDiseaseorAttack"
]

lifestyle_features = [
    f for f in ordered_features
    if f not in basic_features + history_features
]

with st.form("input_form"):

    # =====================
    # 👤 基本情報
    # =====================
    st.markdown("### 👤 基本情報")
    col1, col2 = st.columns(2)

    for i, feature in enumerate(basic_features):
        label = feature_labels.get(feature, feature)
        col = col1 if i % 2 == 0 else col2

        with col:
            if feature == "Sex":
                choice = st.selectbox(label, ["女性", "男性"])
                inputs[feature] = 0 if choice == "女性" else 1

            elif feature == "Age":
                choice = st.selectbox(label, list(age_options.values()))
                inputs[feature] = [k for k, v in age_options.items() if v == choice][0]

            elif feature == "Income":
                choice = st.selectbox(label, list(income_options.values()))
                inputs[feature] = list(income_options.keys())[list(income_options.values()).index(choice)]

            else:
                inputs[feature] = st.number_input(label, min_value=0.0, step=1.0)

    # =====================
    # 🏥 既往歴
    # =====================
    st.markdown("### 🏥 既往歴")
    col1, col2 = st.columns(2)

    for i, feature in enumerate(history_features):
        label = feature_labels.get(feature, feature)
        col = col1 if i % 2 == 0 else col2

        with col:
            choice = st.selectbox(label, ["いいえ", "はい"])
            inputs[feature] = 0 if choice == "いいえ" else 1

    # =====================
    # 🏃 生活習慣
    # =====================
    st.markdown("### 🏃 生活習慣")
    col1, col2 = st.columns(2)

    for i, feature in enumerate(lifestyle_features):
        label = feature_labels.get(feature, feature)
        col = col1 if i % 2 == 0 else col2

        with col:
            if feature == "GenHlth":
                choice = st.selectbox(label, list(genhlth_options.values()))
                inputs[feature] = list(genhlth_options.keys())[list(genhlth_options.values()).index(choice)]

            elif feature == "MentHlth":
                inputs[feature] = st.selectbox(label, list(range(0, 31)))

            elif feature == "PhysHlth":
                inputs[feature] = st.selectbox(label, list(range(0, 31)))

            elif feature in [
                "Smoker","PhysActivity","HvyAlcoholConsump",
                "DiffWalk","Fruits","Veggies"
            ]:
                choice = st.selectbox(label, ["いいえ", "はい"])
                inputs[feature] = 0 if choice == "いいえ" else 1

            else:
                inputs[feature] = st.number_input(label, min_value=0.0, step=1.0)

    submitted = st.form_submit_button(
        "糖尿病リスクを確認する",
        use_container_width=True
    )


#リスク算出
if submitted:
    with st.spinner("データを処理中です..."):
        time.sleep(2)

        inputs["CholCheck"] = 1
        inputs["AnyHealthcare"] = 1
        inputs["NoDocbcCost"] = 0

        x = np.array([inputs[f] for f in feature_names]).reshape(1, -1)

        prob = model.predict_proba(x)[0][1]

        # SHAP計算（影響が大きい特徴量抽出）
        explainer = shap.TreeExplainer(model)
        shap_result = explainer(x)
        values = np.array(shap_result.values)

        if values.ndim == 3:
            shap_vals = values[0, :, 1]
        else:
            shap_vals = values[0]

        df_shap = pd.DataFrame({
            "feature": feature_names,
            "impact": shap_vals
        })

        #行動では変えられない特徴量は除く
        exclude_features = ["Age", "Sex", "Income"]
        df_shap = df_shap[~df_shap["feature"].isin(exclude_features)]
        df_shap = df_shap.sort_values("impact", key=np.abs, ascending=False)

        suppress_factors = []
        increase_factors = []

        for _, row in df_shap.head(3).iterrows():
            label = feature_labels.get(row["feature"], row["feature"])
            if row["impact"] < 0:
                suppress_factors.append(label)
            else:
                increase_factors.append(label)

        # session_state に保存
        st.session_state["predicted"] = True
        st.session_state["prob"] = prob
        st.session_state["suppress_factors"] = suppress_factors
        st.session_state["increase_factors"] = increase_factors

#リスクの状況に応じて、リスクを押し下げている、押し上げている特徴量を特定する
if st.session_state["predicted"] and st.session_state["prob"] is not None:
    prob = st.session_state["prob"]
    suppress_factors = st.session_state["suppress_factors"]
    increase_factors = st.session_state["increase_factors"]

    # ---- リスク分類 ----
    st.write("")
    st.write("")
    st.markdown("### 📊 判定結果")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            label="糖尿病発症リスク",
            value=f"{prob*100:.1f}％"
        )

    with col2:
        if prob < 0.10:
            st.success("🟢 糖尿病リスクは低めです")
        elif prob < 0.30:
            st.warning("🟡 糖尿病リスクがやや高めです")
        else:
            st.error("🔴 糖尿病リスクが高めです")

    st.caption("※ 過去の統計データに基づき、同じ状態の方が糖尿病を発症している確率です")


    # --------------------
    # リスク要因の表示（ハイリスク、ローリスクで表示を出し分ける）
    # --------------------
    def load_css_life():
        st.markdown("""
        <style>
        .tag-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
            margin-bottom: 8px;
        }

        .tag {
            display: inline-block;
            background-color: #eaf2fb;   /* 淡い青 */
            color: #1f4fd8;              /* 青文字 */
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 14px;
            line-height: 1.2;
            white-space: nowrap;
        }
        </style>
        """, unsafe_allow_html=True)


    load_css_life()

    st.markdown("### 🔍 関係している生活習慣")

    with st.container():

        if prob < 0.10 and suppress_factors:

            #ローリスク
            tags_html = "".join(
                [f'<span class="tag">{factor}</span>' for factor in suppress_factors]
            )

            st.markdown(
                f"""
                <div style="border:1px solid rgba(0,0,0,0.1); padding:1em; border-radius:6px;">
                    <div class="tag-container">
                        {tags_html}
                    </div>
                    <p>
                        これらは、糖尿病リスクを低めに保つことに関係している可能性がある要因です。<br>
                        今後も状態を維持することで、現在の評価が保つことができる可能性があります。
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        elif prob >= 0.10 and increase_factors:

            #ハイリスク
            tags_html = "".join(
                [f'<span class="tag">{factor}</span>' for factor in increase_factors]
            )

            st.markdown(
                f"""
                <div style="border:1px solid rgba(0,0,0,0.1); padding:1em; border-radius:6px;">
                    <div class="tag-container">
                        {tags_html}
                    </div>
                    <p>
                        これらは、糖尿病リスクに関係する可能性がある要因です。<br>
                        状態を見直すことで、現在の評価が変わる可能性があります。
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


    #リスクに基づくプロンプトの定義
    if prob < 0.10:
        target_factors = suppress_factors
        advice_mode = "maintain"
        intro_text = (
            "現在の生活習慣で、糖尿病リスクは低めに保たれています。"
            "今の生活を続けるためのヒントを提案してください。"
        )
    else:
        target_factors = increase_factors
        advice_mode = "improve"
        intro_text = (
            "糖尿病リスクが高い状態です。関係する可能性がある生活習慣について、"
            "改善のヒントを提案してください。"
        )

    st.session_state["intro_text"] = intro_text
    st.session_state["target_factors"] = target_factors


    # --------------------
    # Gemini設定
    # --------------------
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    genai_model = genai.GenerativeModel("gemini-2.5-flash")
    
    st.markdown("---")
    st.markdown("### 💡結果をもとに、生活のヒントを確認できます")


#PDF読み込み
pdf_path = "tokyo-advice.pdf"

import pdfplumber

@st.cache_data
def load_pdf_pages(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                pages.append({
                    "text": text,
                    "page": i,
                    "source": pdf_path
                })
    return pages

#PDFをテキスト化してリトリーバルの準備
def split_text_with_meta(pages, chunk_size=500, overlap=100):
    chunks = []
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({
                "text": text[start:end],
                "page": p["page"],
                "source": p["source"]
            })
            start = end - overlap
    return chunks

@st.cache_data
def prepare_vectorstore(pdf_path):
    pages = load_pdf_pages(pdf_path)
    chunks = split_text_with_meta(pages)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([c["text"] for c in chunks])
    return embeddings, chunks, model

embeddings, chunks, embed_model = prepare_vectorstore(pdf_path)

def retrieve_context(query, embeddings, chunks, model, top_k=3):
    query_vec = model.encode([query])
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    results = []
    for i in top_indices:
        results.append({
            "text": chunks[i]["text"],
            "page": chunks[i]["page"],
            "source": chunks[i]["source"],
            "score": sims[i]
        })
    return results

#RAGを実行してアドバイスを生成する
#関係ある特徴量等を考慮して個別最適化された内容を表示
if st.session_state.get("predicted", False):

    if st.button("アドバイスを見る", use_container_width=True):
        with st.spinner("アドバイスを作成しています..."):
            time.sleep(1)

            query = f"""
            状況: {st.session_state["intro_text"]}
            関連要因: {'、'.join(st.session_state["target_factors"])}
            """

            retrieved_results = retrieve_context(
                query,
                embeddings,
                chunks,
                embed_model
            )

            reference_text = "\n\n".join([
                f"【出典】{r['source']} / p.{r['page']}\n{r['text']}"
                for r in retrieved_results
            ])

            prompt = f"""
このアプリは対象者の糖尿病リスクを予測して表示するものです。
あなたは対象者のリスクや状況に基づき、保健師の立場でアドバイスを行います。

【対象者の状況】
{st.session_state["prob"]}
{st.session_state["intro_text"]}


【リスクに関係する要因】
・{'、'.join(st.session_state["target_factors"])}

【条件】
・結果についての言及は行わない
・診断や治療の指示は行わない
・日常生活で無理なく取り入れやすい行動に限定する
・各行動について、行った場合に期待される変化を必ず記載する
・3個の箇条書きで出力する
・行動案は太字で1文とする
・行動案の次の行で「期待される変化：」を付けて記載する
・専門用語は使わない

【条件】
    ・結果についての言及は行わない
    ・診断や治療の指示は行わない
    ・日常生活で無理なく取り入れやすい行動に限定する
    ・各行動について、行った場合に期待される変化を必ず記載する
    ・3個の箇条書きで出力する
    ・行動案は太字で1文とする
    ・行動案の次の行で「期待される変化：」を付けて記載する
    ・1文は簡潔で、専門用語は使わない

    【出力形式】
    以下のフォーマットを厳密に守ること。

    ・出力はMarkdown形式とする
    ・最初に1行のフィードバックメッセージを表示する
    ・その後、必ず以下の形式で3つ出力する

    **メッセージ（1文）**

    - **行動案（1文）**  
    期待される変化：◯◯◯

    - **行動案（1文）**  
    期待される変化：◯◯◯

    - **行動案（1文）**  
    期待される変化：◯◯◯

    【参考資料】
    {reference_text}
    """

            response = genai_model.generate_content(prompt)

            with st.container():
                st.markdown(
                    f"""
                    <div style="border:1px solid rgba(0,0,0,0.1); padding:1em; border-radius:6px;">
                        {response.text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            #参照した部分を表示する
            REFERENCE_TITLE = "糖尿病発症予防ガイドブック「今日から予防！糖尿病」"
            REFERENCE_URL = "https://www.hokeniryo1.metro.tokyo.lg.jp/kensui/tonyo/citizen/6leaflet.html"

            # ページ番号のみ重複排除して昇順にする
            pages = sorted({r["page"] for r in retrieved_results})

            with st.expander("参照ページ"):
                st.markdown(
                    f"**{REFERENCE_TITLE}** "
                    f"[資料リンク]({REFERENCE_URL})"
                )
                st.markdown(
                    "参照ページ：" + "、".join([f"p.{p}" for p in pages])
                )