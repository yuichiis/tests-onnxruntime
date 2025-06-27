# coding: utf-8
from transformers import AutoTokenizer # AutoTokenizer は引き続き使用
import onnxruntime
import numpy as np
# hf_hub_download をインポート
from huggingface_hub import hf_hub_download #, HfHubDownloadError
import os
import time
import json

# --- 設定 ---
model_hub_id = "onnx-community/gemma-3-1b-it-ONNX"
tokenizer_hub_id = "google/gemma-3-1b-it"
# ダウンロードしたいONNXモデルファイル名を指定 (onnxサブディレクトリ内)
# onnx_model_repo_filename = "onnx/model_int8.onnx" # ← これを指定
# onnx_model_repo_filename = "onnx/model_fp16.onnx"
onnx_model_repo_filename = "onnx/model.onnx"

# 対応する .onnx_data ファイル名 (model.onnx の場合のみ考慮)
onnx_data_repo_filename = "onnx/model.onnx_data" if onnx_model_repo_filename == "onnx/model.onnx" else None

config_repo_filename = "config.json" # 設定ファイル

max_new_tokens = 300

# --- 1. 必要なファイルのダウンロードと設定/トークナイザー/ONNXセッションのロード ---
print("必要なファイルをダウンロード中...")
try:
    # config.json をダウンロード
    print(f"- {config_repo_filename}")
    local_config_path = hf_hub_download(
        repo_id=model_hub_id,
        filename=config_repo_filename
    )
    print(f"  ローカルパス: {local_config_path}")

    # 指定された ONNX モデルファイルをダウンロード
    print(f"- {onnx_model_repo_filename}")
    local_onnx_path = hf_hub_download(
        repo_id=model_hub_id,
        filename=onnx_model_repo_filename
    )
    print(f"  ローカルパス: {local_onnx_path}")

    # 必要なら .onnx_data ファイルもダウンロード
    local_onnx_data_path = None
    if onnx_data_repo_filename:
        try:
            print(f"- {onnx_data_repo_filename}")
            local_onnx_data_path = hf_hub_download(
                repo_id=model_hub_id,
                filename=onnx_data_repo_filename
            )
            print(f"  ローカルパス: {local_onnx_data_path}")
            # 重要: .onnx と .onnx_data は同じディレクトリにある必要があるが、
            # hf_hub_download は通常キャッシュ内の同じ場所に配置してくれるはず。
        #except HfHubDownloadError as e:
        #    # .onnx_data が存在しない場合のエラーは無視しても良い場合がある
        #    # (モデルが小さい場合など) が、基本的には必要。
        #    print(f"警告: {onnx_data_repo_filename} のダウンロードに失敗しました: {e}")
        #    print("大きなモデルの場合、これが原因でロードに失敗する可能性があります。")
        except FileNotFoundError as e:
            print(f"ファイルが見つかりません: {e}")
            exit()
        except Exception as e:
            print(f"ファイルのダウンロードまたは初期化中にエラーが発生しました: {e}")
            print("リポジトリIDやファイル名、ネットワーク接続、ライブラリのバージョンを確認してください。")
            exit()

    print("ファイルダウンロード完了。")

    print("設定ファイル(直接読み込み)とトークナイザーをロード中...")
    # config.json を直接読み込む (ダウンロードしたローカルパスを使う)
    with open(local_config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    # 必要な設定値を取得
    try:
        num_key_value_heads = config_data['num_key_value_heads']
        head_dim = config_data['head_dim']
        num_hidden_layers = config_data['num_hidden_layers']
    except KeyError as e:
        print(f"エラー: config.jsonに必要なキーが見つかりません: {e}")
        exit()

    # tokenizer は通常通りロード (必要なファイルは内部でダウンロードされる)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_hub_id, trust_remote_code=True)
    print("設定(直接読み込み)とトークナイザーのロード完了。")

    print(f"ONNX Runtime セッションをロード中 ({local_onnx_path})...")
    providers = [
        ('DmlExecutionProvider', {'device_id': 0}),
        'CPUExecutionProvider'
    ]
    # providers = ['CPUExecutionProvider'] # CPUで試す場合

    session_options = onnxruntime.SessionOptions()
    # ONNXモデルのローカルパスを指定
    decoder_session = onnxruntime.InferenceSession(
        local_onnx_path, # ダウンロードした .onnx ファイルのパス
        providers=providers,
        sess_options=session_options
    )
    print(f"実際に使用されたExecution Provider: {decoder_session.get_providers()}")
    print("ONNX Runtime セッションのロード完了。")

#except HfHubDownloadError as e:
#     print(f"ファイルのダウンロードに失敗しました: {e}")
#     print("リポジトリIDやファイル名が正しいか、ネットワーク接続を確認してください。")
#     exit()
except FileNotFoundError as e:
     print(f"ファイルが見つかりません: {e}")
     exit()
except Exception as e:
    print(f"初期化中にエラーが発生しました: {e}")
    exit()


# --- (コード省略) ---
## 設定値の取得とKVキャッシュデータ型の決定
try:
    eos_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if eos_token_id == tokenizer.unk_token_id:
        print("警告: トークナイザーが '<end_of_turn>' を認識できません。EOS ID=1 (<eos>) を使用します。")
        eos_token_id = tokenizer.eos_token_id
    print("KVキャッシュのデータ型を強制的に float32 (<class 'numpy.float32'>) に設定します。")
    kv_cache_dtype = np.float32
    print(f"KV Heads: {num_key_value_heads}, Head Dim: {head_dim}, Layers: {num_hidden_layers}, EOS ID: {eos_token_id}, KV Cache dtype: {kv_cache_dtype}")
except ValueError as e:
    print(f"エラー: '<end_of_turn>' トークンの取得に失敗しました: {e}")
    exit()

# --- 2. 入力データの準備 ---
messages = [
  {"role": "user", "content": "日本の首都とその有名な観光地を3つ教えてください。"}
]
print("\n--- プロンプト ---")
prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(prompt_str)
print("------------------")
inputs = tokenizer(prompt_str, return_tensors="np")
input_ids = inputs['input_ids'].astype(np.int64)
batch_size = input_ids.shape[0]
seq_len = input_ids.shape[1]

# 初期KVキャッシュ
past_key_values = {}
for layer in range(num_hidden_layers):
    past_key_values[f'past_key_values.{layer}.key'] = np.zeros(
        [batch_size, num_key_value_heads, 0, head_dim], dtype=kv_cache_dtype
    )
    past_key_values[f'past_key_values.{layer}.value'] = np.zeros(
        [batch_size, num_key_value_heads, 0, head_dim], dtype=kv_cache_dtype
    )

# 初期 position_ids
position_ids = np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len)
print(f"Initial input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}")
print(f"Initial position_ids shape: {position_ids.shape}, dtype: {position_ids.dtype}")
print(f"Initial KV cache key example: past_key_values.0.key, shape: {past_key_values['past_key_values.0.key'].shape}, dtype: {past_key_values['past_key_values.0.key'].dtype}")


# --- 3. 生成ループ ---
print("\nテキスト生成を実行中...")
start_time = time.time()
generated_token_ids = []

for i in range(max_new_tokens):
    model_inputs = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        **past_key_values
    }
    try:
        outputs = decoder_session.run(None, model_inputs)
    except Exception as e:
        print(f"\nONNX推論中にエラーが発生しました: {e}")
        print("入力形状やデータ型、モデルの入力/出力名を確認してください。")
        print("Model Inputs provided:", {k: f"{v.shape}, {v.dtype}" for k, v in model_inputs.items() if 'past' not in k})
        break

    logits = outputs[0]
    present_key_values_flat = outputs[1:]
    next_token_logits = logits[:, -1, :]
    next_token_id_array = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
    next_token_id = next_token_id_array.item()
    generated_token_ids.append(next_token_id)

    if next_token_id == eos_token_id:
        print("\nEOSトークンを検出しました。")
        break

    print(tokenizer.decode([next_token_id]), end='', flush=True)

    input_ids = next_token_id_array
    position_ids = np.array([[position_ids[:, -1].item() + 1]], dtype=np.int64)

    if len(present_key_values_flat) != len(past_key_values):
        print(f"\nエラー: KVキャッシュの数が一致しません。Expected {len(past_key_values)}, Got {len(present_key_values_flat)}")
        break
    keys = list(past_key_values.keys())
    for j in range(0, len(present_key_values_flat)):
        past_key_values[keys[j]] = present_key_values_flat[j]

else:
    print("\n最大生成トークン数に達しました。")

end_time = time.time()
print(f"\n生成完了 ({end_time - start_time:.2f}秒)")

# --- 4. 最終結果の出力 ---
print("\n--- 生成結果 (デコード) ---")
final_text = tokenizer.decode(generated_token_ids)
print(final_text)
print("--------------------------")
