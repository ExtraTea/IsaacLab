#!/usr/bin/env python
# filepath: /home/daisuke/IsaacLab/scripts/tools/simple_usdcat.py

"""
USDファイルを簡単に扱うための単純なスクリプト
"""

import argparse
import os
import sys
import subprocess

def convert_to_ascii(input_path, output_path=None, max_lines=0):
    """
    USDファイルをascii形式に変換する（外部コマンド使用）
    
    Args:
        input_path (str): 入力USDファイルのパス
        output_path (str, optional): 出力ファイルのパス。指定しない場合は標準出力に表示
        max_lines (int, optional): 出力する最大行数。0の場合は制限なし
    """
    # 入力ファイルのチェック
    if not os.path.exists(input_path):
        print(f"エラー: ファイルが見つかりません: {input_path}", file=sys.stderr)
        return False
    
    try:
        # USD標準ツールのusdcatコマンドを使用
        cmd = ['usdcat', input_path]
        
        if output_path:
            cmd.extend(['--out', output_path])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"エラー: usdcatの実行に失敗しました: {result.stderr}", file=sys.stderr)
            return False
        
        if output_path:
            print(f"出力ファイルを保存しました: {output_path}")
        else:
            # 出力行数の制限（もしあれば）
            content = result.stdout
            if max_lines > 0:
                lines = content.split('\n')
                limited_content = '\n'.join(lines[:max_lines])
                print(limited_content)
                print(f"\n... 出力を{max_lines}行に制限しました（全{len(lines)}行）...")
            else:
                print(content)
        
        return True
        
    except Exception as e:
        print(f"エラー: 変換中に問題が発生しました: {str(e)}", file=sys.stderr)
        return False


def print_file_info(input_path):
    """
    USDファイルの基本情報を表示する
    
    Args:
        input_path (str): 入力USDファイルのパス
    """
    # 入力ファイルのチェック
    if not os.path.exists(input_path):
        print(f"エラー: ファイルが見つかりません: {input_path}", file=sys.stderr)
        return False
    
    try:
        # ファイルサイズと修正日時
        file_size = os.path.getsize(input_path)
        file_mtime = os.path.getmtime(input_path)
        
        # ファイル情報
        print(f"USDファイル: {input_path}")
        print(f"サイズ: {file_size / (1024*1024):.2f} MB")
        print(f"更新日時: {os.ctime(file_mtime)}")
        
        # ファイルタイプを確認（拡張子から）
        _, ext = os.path.splitext(input_path)
        if ext.lower() == '.usd':
            print("フォーマット: バイナリUSD (.usd)")
        elif ext.lower() == '.usda':
            print("フォーマット: ASCIIテキストUSD (.usda)")
        elif ext.lower() == '.usdc':
            print("フォーマット: コンパイル済みUSD (.usdc)")
        elif ext.lower() == '.usdz':
            print("フォーマット: 圧縮USDBパッケージ (.usdz)")
        else:
            print(f"フォーマット: 不明 ({ext})")
        
        return True
        
    except Exception as e:
        print(f"エラー: 分析中に問題が発生しました: {str(e)}", file=sys.stderr)
        return False


def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='USDファイルをascii形式で出力するツール')
    parser.add_argument('input_file', help='入力USDファイルのパス')
    parser.add_argument('-o', '--output', help='出力ファイルのパス（指定しない場合は標準出力）')
    parser.add_argument('--info', action='store_true', help='USDファイルの情報のみを表示')
    parser.add_argument('--max-lines', type=int, default=100, help='出力する最大行数（0=制限なし）')
    args = parser.parse_args()
    
    # ホームディレクトリの展開（~/を絶対パスに変換）
    input_path = os.path.expanduser(args.input_file)
    output_path = os.path.expanduser(args.output) if args.output else None
    
    # 実行
    if args.info:
        print_file_info(input_path)
    else:
        success = convert_to_ascii(input_path, output_path, max_lines=args.max_lines)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
