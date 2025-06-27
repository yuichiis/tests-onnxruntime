#### DOWNLOAD #############
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch # PyTorchテンソルで結果を確認する場合など (必須ではない)
import numpy as np # ONNX RuntimeはNumPy配列を扱うことが多い

# モデルIDを指定 (Hub上のONNX最適化済みモデル or 元のPyTorch/TFモデル)
# この例ではoptimumによって最適化・変換されたONNXモデルを指定
model_id = "optimum/distilbert-base-uncased-finetuned-sst-2-english"
# もしくは元のPyTorchモデルIDを指定しても、optimumがONNXに変換してくれる
# model_id = "distilbert-base-uncased-finetuned-sst-2-english"

# ONNXモデルをロード
# providerを指定して実行環境 (CPU, CUDA, TensorRTなど) を選べる
# デフォルトは 'CPUExecutionProvider'
model = ORTModelForSequenceClassification.from_pretrained(model_id)
# GPUを使いたい場合:
# model = ORTModelForSequenceClassification.from_pretrained(model_id, provider="CUDAExecutionProvider")

# 対応するトークナイザーをロード (通常は元のモデルと同じIDでOK)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("ONNXモデルとトークナイザーのロード完了")

#### preparing input data #############
text = "Using ONNX models with Hugging Face is quite easy!"

# テキストをトークナイズし、NumPy配列でテンソルを取得
inputs = tokenizer(text, return_tensors="np")

print("入力データ:")
print(inputs)

### predicting ######################

# モデルに入力データを渡して推論を実行
outputs = model(**inputs)

# 出力を確認 (通常、logitsなどが含まれる)
logits = outputs.logits
print("\n推論結果 (logits):")
print(logits)

##### outputs ###################
# 最も確率の高いクラスのIDを取得
predicted_class_id = np.argmax(logits, axis=-1)[0]

# モデルのconfigからIDとラベルのマッピングを取得して表示
label = model.config.id2label[predicted_class_id]

# ソフトマックス関数で確率に変換 (オプション)
probabilities = torch.softmax(torch.from_numpy(logits), dim=-1)[0].numpy()
probability = probabilities[predicted_class_id]


print(f"\n入力テキスト: {text}")
print(f"予測ラベルID: {predicted_class_id}")
print(f"予測ラベル: {label}")
print(f"予測確率: {probability:.4f}")

# 他のテキストでも試す
text_neg = "I didn't like the performance."
inputs_neg = tokenizer(text_neg, return_tensors="np")
outputs_neg = model(**inputs_neg)
logits_neg = outputs_neg.logits
predicted_class_id_neg = np.argmax(logits_neg, axis=-1)[0]
label_neg = model.config.id2label[predicted_class_id_neg]
print(f"\n入力テキスト: {text_neg}")
print(f"予測ラベル: {label_neg}")

