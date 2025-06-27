# coding: utf-8
# from transformers import AutoConfig, AutoTokenizer # AutoConfig は不要になる
from transformers import AutoTokenizer # AutoTokenizer は引き続き使用
import onnxruntime
import numpy as np
from huggingface_hub import snapshot_download
import os
import time
import json # json ライブラリをインポート

# --- 設定 ---
model_hub_id = "onnx-community/gemma-3-1b-it-ONNX"
tokenizer_hub_id = "google/gemma-3-1b-it"
onnx_model_filename = "model.onnx" # または他のファイル
#onnx_model_filename = "model_fp16.onnx" # または他のファイル
#onnx_model_filename = "model_int8.onnx" # または他のファイル
#onnx_model_filename = "model_uint8.onnx" # または他のファイル
#onnx_model_filename = "model_q4.onnx" # または他のファイル
#onnx_model_filename = "model_q4f16.onnx" # または他のファイル
#max_new_tokens = 150
max_new_tokens = 300

# --- 1. モデルパスの取得と設定/トークナイザー/ONNXセッションのロード ---
print("モデルファイルをダウンロード/確認中...")
try:
    local_model_path = snapshot_download(model_hub_id)
    print(f"ローカルのリポジトリルート: {local_model_path}")

    onnx_file_path = os.path.join(local_model_path, "onnx", onnx_model_filename)
    onnx_data_path = os.path.join(local_model_path, "onnx", "model.onnx_data")
    config_path = os.path.join(local_model_path, "config.json") # config.json のパス

    # 必要なファイルの存在チェック
    if not os.path.exists(config_path):
        print(f"エラー: config.json が見つかりません: {config_path}")
        exit()
    if not os.path.exists(onnx_file_path):
        print(f"エラー: 指定されたONNXファイルが見つかりません: {onnx_file_path}")
        exit()
    # (model.onnx_data の警告は省略)

    print("設定ファイル(直接読み込み)とトークナイザーをロード中...")
    # --- ★ここから修正★ ---
    # config = AutoConfig.from_pretrained(local_model_path) # ← この行を削除

    # config.json を直接読み込む
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    # 必要な設定値を取得
    try:
        num_key_value_heads = config_data['num_key_value_heads']
        head_dim = config_data['head_dim']
        num_hidden_layers = config_data['num_hidden_layers']
        # model_type = config_data['model_type'] # 参考: これが 'gemma3_text'
    except KeyError as e:
        print(f"エラー: config.jsonに必要なキーが見つかりません: {e}")
        print("config.jsonの内容を確認してください:", config_path)
        exit()
    # --- ★修正ここまで★ ---

    # tokenizer はそのままロード
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_hub_id, trust_remote_code=True)
    print("設定(直接読み込み)とトークナイザーのロード完了。")

    print(f"ONNX Runtime セッションをロード中 ({onnx_file_path})...")
    # 実行プロバイダーを指定 (DirectML を優先し、ダメならCPUへフォールバック)
    providers = [
        ('DmlExecutionProvider', {
            'device_id': 0 # 複数のGPUがある場合、通常は0でデフォルトGPU
            # 'enable_metacommands': True # パフォーマンス向上の可能性 (オプション)
        }),
        'CPUExecutionProvider' # DirectMLが使えない場合のフォールバック
    ]
    #providers = ['CPUExecutionProvider'] # CPUのみに強制
    # 複数のGPUがあるか不明、またはデフォルトで良ければシンプルに書くことも可能
    # providers = ['DmlExecutionProvider', 'CPUExecutionProvider']    
    session_options = onnxruntime.SessionOptions()
    #session_options.log_severity_level = 0 # 0: Verbose, 1: Info, 2: Warning, 3: Error
    decoder_session = onnxruntime.InferenceSession(
        onnx_file_path,
        providers=providers,
        sess_options=session_options
    )
    # 実際に使用されたプロバイダーを確認 (デバッグ用)
    print(f"実際に使用されたExecution Provider: {decoder_session.get_providers()}")    
    print("ONNX Runtime セッションのロード完了。")

except Exception as e:
    print(f"初期化中にエラーが発生しました: {e}")
    # DirectML固有のエラーが出る可能性もある
    print("DirectMLの初期化に失敗した可能性があります。グラフィックドライバが最新か、OSバージョンを確認してください。")    
    exit()


## 設定値の取得とKVキャッシュデータ型の決定 (取得部分は上に移動済み)
try:
    # EOSトークンID
    eos_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if eos_token_id == tokenizer.unk_token_id:
        print("警告: トークナイザーが '<end_of_turn>' を認識できません。EOS ID=1 (<eos>) を使用します。")
        eos_token_id = tokenizer.eos_token_id # Gemma v2/v3 のデフォルトEOS IDは 1

    # KVキャッシュのデータ型を決定
    # モデルファイル名に関わらず、強制的に float32 を試す
    print(f"KVキャッシュのデータ型を強制的に float32 ({np.float32}) に設定します。")
    kv_cache_dtype = np.float32
    # 以下の自動判定ロジックはコメントアウトまたは削除
    # kv_cache_dtype = np.float32
    # if "fp16" in onnx_model_filename.lower():
    #     print("FP16モデルを検出。KVキャッシュのデータ型を float16 に設定します。")
    #     kv_cache_dtype = np.float16
    # elif "int8" in onnx_model_filename.lower() or "quantized" in onnx_model_filename.lower():
    #     print("量子化モデルを検出。KVキャッシュのデータ型を float16 に設定します。")
    #     kv_cache_dtype = np.float16

    print(f"KV Heads: {num_key_value_heads}, Head Dim: {head_dim}, Layers: {num_hidden_layers}, EOS ID: {eos_token_id}, KV Cache dtype: {kv_cache_dtype}")

except ValueError as e:
    print(f"エラー: '<end_of_turn>' トークンの取得に失敗しました: {e}")
    exit()


# --- 2. 入力データの準備 ---
# (変更なし)
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
# (変更なし、ただしエラーチェックは重要)
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
        print("Model Inputs provided:", {k: f"{v.shape}, {v.dtype}" for k, v in model_inputs.items() if 'past' not in k}) # KVキャッシュ以外を表示
        # KVキャッシュの形状も確認したい場合
        # print("KV Cache shapes:", {k: v.shape for k, v in model_inputs.items() if 'past' in k})
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
# (変更なし)
print("\n--- 生成結果 (デコード) ---")
final_text = tokenizer.decode(generated_token_ids)
print(final_text)
print("--------------------------")
