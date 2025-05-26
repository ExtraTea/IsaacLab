#!/usr/bin/env python
# filepath: /home/daisuke/IsaacLab/scripts/tools/usdcat.py

"""
USDファイルをascii形式で出力するスクリプト
"""

import argparse
import os
import sys
from pxr import Usd, UsdGeom, UsdUtils


def convert_to_ascii(input_path, output_path=None, max_lines=0):
    """
    USDファイルをascii形式に変換する
    
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
        # パッケージで正しいUSDステージオープン関数を見つける
        try:
            # 方法1: pxrのUsdモジュールから直接
            stage = Usd.Stage.Open(input_path)
        except AttributeError:
            try:
                # 方法2: UsdStage関数を使用
                stage = Usd.UsdStage.Open(input_path)
            except AttributeError:
                # 方法3: 必要なものを明示的にインポート
                from pxr.Usd import Stage
                stage = Stage.Open(input_path)
        
        if not stage:
            print(f"エラー: USDファイルを開けませんでした: {input_path}", file=sys.stderr)
            return False

        # ExportToString でascii形式のUSDを取得
        ascii_content = stage.ExportToString()
        
        # 出力先の処理
        if output_path:
            with open(output_path, 'w') as f:
                f.write(ascii_content)
            print(f"出力ファイルを保存しました: {output_path}")
        else:
            # 行数制限がある場合は、その分だけ出力
            if max_lines > 0:
                lines = ascii_content.split('\n')
                limited_content = '\n'.join(lines[:max_lines])
                print(limited_content)
                print(f"\n... 出力を{max_lines}行に制限しました（全{len(lines)}行）...")
            else:
                print(ascii_content)
        
        return True
    
    except Exception as e:
        print(f"エラー: 変換中に問題が発生しました: {str(e)}", file=sys.stderr)
        return False


def print_usd_summary(input_path):
    """
    USDファイルの概要情報を表示する
    
    Args:
        input_path (str): 入力USDファイルのパス
    """
    if not os.path.exists(input_path):
        print(f"エラー: ファイルが見つかりません: {input_path}", file=sys.stderr)
        return False
    
    try:
        # USDステージを開く
        stage = Usd.Stage.Open(input_path)
        if not stage:
            print(f"エラー: USDファイルを開けませんでした: {input_path}", file=sys.stderr)
            return False
        
        # ファイル情報
        print(f"USDファイル: {input_path}")
        print(f"アップアクシス: {UsdGeom.GetStageUpAxis(stage)}")
        print(f"メートル単位: {UsdGeom.GetStageMetersPerUnit(stage)}")
        
        # ルートレイヤーの取得
        root_layer = stage.GetRootLayer()
        print(f"レイヤー数: {len(stage.GetLayerStack())}")
        
        # プリム（オブジェクト）情報
        prims = list(stage.Traverse())
        print(f"プリム数: {len(prims)}")
        
        # プリムタイプの分類
        prim_types = {}
        for prim in prims:
            prim_type = prim.GetTypeName()
            if prim_type in prim_types:
                prim_types[prim_type] += 1
            else:
                prim_types[prim_type] = 1
        
        print("\nプリムタイプの内訳:")
        for prim_type, count in prim_types.items():
            print(f"  {prim_type}: {count}")
        
        # メッシュ情報
        mesh_prims = [p for p in prims if p.GetTypeName() == "Mesh"]
        if mesh_prims:
            print("\nメッシュ情報:")
            for i, mesh_prim in enumerate(mesh_prims[:5]):  # 最初の5つのメッシュのみ表示
                mesh = UsdGeom.Mesh(mesh_prim)
                points_attr = mesh.GetPointsAttr()
                face_count_attr = mesh.GetFaceVertexCountsAttr()
                faces_attr = mesh.GetFaceVertexIndicesAttr()
                
                point_count = len(points_attr.Get()) if points_attr.Get() else 0
                face_count = len(face_count_attr.Get()) if face_count_attr.Get() else 0
                
                print(f"  メッシュ {i+1}: {mesh_prim.GetPath()}")
                print(f"    頂点数: {point_count}")
                print(f"    面数: {face_count}")
            
            if len(mesh_prims) > 5:
                print(f"  ... 他 {len(mesh_prims) - 5} 個のメッシュ")
        
        return True
    
    except Exception as e:
        print(f"エラー: 分析中に問題が発生しました: {str(e)}", file=sys.stderr)
        return False


def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='USDファイルをascii形式で出力するツール')
    parser.add_argument('input_file', help='入力USDファイルのパス')
    parser.add_argument('-o', '--output', help='出力ファイルのパス（指定しない場合は標準出力）')
    parser.add_argument('--summary', action='store_true', help='USDファイルの概要のみを表示（詳細な頂点データなどは省略）')
    parser.add_argument('--max-lines', type=int, default=0, help='出力する最大行数（0=制限なし）')
    args = parser.parse_args()
    
    # ホームディレクトリの展開（~/を絶対パスに変換）
    input_path = os.path.expanduser(args.input_file)
    output_path = os.path.expanduser(args.output) if args.output else None
    
    # 変換実行
    if args.summary:
        print_usd_summary(input_path)
        return 0
    else:
        success = convert_to_ascii(input_path, output_path, max_lines=args.max_lines)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())