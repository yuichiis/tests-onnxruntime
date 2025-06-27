import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig # AutoConfigを追加
from optimum.onnxruntime import ORTModelForCausalLM
import time
from huggingface_hub import snapshot_download
import os

# 1. モデルIDとトークナイザーIDの設定
model_id = "onnx-community/gemma-3-1b-it-ONNX"
tokenizer_id = "google/gemma-3-1b-it"

# 2. トークナイザーのロード
print(f"'{tokenizer_id}' からトークナイザーをロード中...")
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    print("トークナイザーのロード完了。")
except Exception as e:
    print(f"トークナイザーのロード中にエラー: {e}")
    exit()

# 3. ONNXモデルのロード (ローカル経由)
print(f"'{model_id}' からモデルファイルをダウンロード/確認中...")
try:
    # snapshot_downloadでリポジトリ全体をローカルにダウンロード/キャッシュ確認
    local_model_repo_path = snapshot_download(model_id)
    print(f"ローカルのリポジトリルート: {local_model_repo_path}")

    # --- ここを修正 ---
    # 使用するONNXファイルへのパスを指定 (onnxサブディレクトリ内)
    # 他の精度 (例: onnx/model_fp16.onnx) を試す場合はここを変更
    onnx_relative_path = os.path.join("onnx", "model.onnx")
    # 絶対パスも念のため作成 (エラーチェック用)
    onnx_absolute_path = os.path.join(local_model_repo_path, onnx_relative_path)
    onnx_data_absolute_path = os.path.join(local_model_repo_path, "onnx", "model.onnx_data")
    config_path = os.path.join(local_model_repo_path, "config.json")


    # 必要なファイルが存在するか確認
    if not os.path.exists(config_path):
        print(f"エラー: config.json が見つかりません: {config_path}")
        exit()
    if not os.path.exists(onnx_absolute_path):
        print(f"エラー: 指定されたONNXファイルが見つかりません: {onnx_absolute_path}")
        exit()
    # model.onnx を使う場合は model.onnx_data も必要
    if os.path.basename(onnx_relative_path) == "model.onnx" and not os.path.exists(onnx_data_absolute_path):
         print(f"エラー: model.onnx に対応する model.onnx_data が見つかりません: {onnx_data_absolute_path}")
         exit()
    # --- 修正ここまで ---


    print(f"'{local_model_repo_path}' から設定を読み込み、ONNXモデル '{onnx_relative_path}' をロード中...")
    # from_pretrainedにリポジトリのルートパスを渡し (config.jsonのため)、
    # file_name で onnx サブディレクトリを含むファイル名を指定する
    model = ORTModelForCausalLM.from_pretrained(
        local_model_repo_path,        # config.json などが含まれるルートパス
        file_name=onnx_relative_path, # 使用するONNXファイルへの相対パス
        provider="CPUExecutionProvider", # または "CUDAExecutionProvider"
        use_cache=True,
        use_io_binding=False
    )
    print("ONNXモデルのロード完了。")

# --- 代替案 (configを別途ロード) ---
#    print(f"'{local_model_repo_path}' から設定をロード中...")
#    config = AutoConfig.from_pretrained(local_model_repo_path)
#    onnx_model_folder = os.path.join(local_model_repo_path, "onnx")
#    print(f"'{onnx_model_folder}' からONNXモデルをロード中...")
#    model = ORTModelForCausalLM.from_pretrained(
#        onnx_model_folder, # ONNXファイルがあるディレクトリ
#        config=config,     # ロードしたconfigオブジェクトを渡す
#        file_name="model.onnx", # onnx_model_folder内のファイル名 (デフォルト)
#        provider="CPUExecutionProvider",
#        use_cache=True,
#        use_io_binding=False
#    )
#    print("ONNXモデルのロード完了。")
# --- 代替案ここまで ---

except Exception as e:
    print(f"モデルのロード中にエラーが発生しました: {e}")
    print("必要なライブラリやモデルファイルのダウンロード状況、ファイルパス、メモリ状況を確認してください。")
    exit()

# --- 以降のコードは変更なし ---

# 4. プロンプトの準備
messages = [
    {"role": "user", "content": "日本の首都とその有名な観光地を3つ教えてください。"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n--- プロンプト ---")
print(prompt)
print("------------------")

# 5. 入力データの準備
inputs = tokenizer(prompt, return_tensors="np")

print("\nテキスト生成を実行中...")
start_time = time.time()

# 6. 推論の実行
try:
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    end_time = time.time()
    print(f"生成完了 ({end_time - start_time:.2f}秒)")

    # 7. 出力のデコード
    generated_ids = outputs[0]
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n--- 生成結果 ---")
    print(decoded_text)
    print("----------------")

except Exception as e:
    print(f"テキスト生成中にエラーが発生しました: {e}")

    