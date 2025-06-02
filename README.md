# RagChatBot Sample001

簡易版のRAG ChatBotです。

ブラウザUI（テキスト入力欄、送信ボタン、回答表示）とデバッグ用のログ機能を備え、Markdownおよびテキストファイル（.md, .txt）を入力として処理する仕様です。

```mermaid
flowchart TD
    A[スクリプト開始時] --> B["文書群（text,markdown,asciidoc）を読み込み"]
    B --> C[テキストを500文字ごとに分割してチャンク化]
    C --> D[SentenceTransformerで文書チャンクをベクトル化]
    D --> E[FAISSでインデックス構築]

    A2[ユーザーが質問を送信] --> F[質問をベクトルに変換]
    F --> G[FAISSで類似文書チャンクを検索]
    G --> H[取得した文書チャンクを送信用メッセージに変換]
    H --> I[OpenAIへメッセージを送信]
    I --> J[OpenAIから回答を取得]

    style A fill:#a8d5a2,stroke:#3a7734,stroke-width:2px,color:#000
    style A2 fill:#a8d5a2,stroke:#3a7734,stroke-width:2px,color:#000
```

実行するには、OpenAIのAPIキーが必要です。OpenAI Platformのアカウントが必要です。（従量課金）

## ファイル

```
ディレクトリ構造
├── documents/ # 入力ドキュメント（AsciiDoc、Markdown、Text）
│ ├── test1.md
│ ├── test2.adoc
│ ├── test3.txt
├── templates/ # HTMLテンプレート
│ ├── index.html # ブラウザUI
├── app.py # FastAPIサーバーコード
├── requirements.txt # 依存ライブラリ
├── .env # 環境変数（OpenAI APIキー）
```
## 準備

私はPython3.11を利用しました。（正確にはWinpython64-3.11.8.0dot.exeを利用しました）

以下で依存ライブラリをインストールします。

```
> python -m pip install -r requirements.txt
```

## 実行

```
> python app.py
```

ブラウザを起動して、`http://127.0.0.1:8000`

## ライセンス

MITライセンスです。  

## 免責事項

本ソフトウェアを使用する際は自己責任でお願いいたします。  
利用により生じた損害等について一切の責任を負いかねます。