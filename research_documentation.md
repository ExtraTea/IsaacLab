# G1ヒューマノイドロボットの粘弾性分解制御と強化学習による歩行動作生成に関する研究ドキュメント

## 1. はじめに

### 研究の目的と概要
本研究は、Unitree G1ヒューマノイドロボットを対象とし、粘弾性分解制御（Resolved Viscoelasticity Control, RVC）の概念を強化学習（Reinforcement Learning, RL）の枠組みに統合することで、ロバストかつ効率的な歩行動作を生成することを目的とする。シミュレーション環境にはNVIDIA Isaac Labを用い、強化学習ライブラリとしてskrlを採用する。本ドキュメントは、関連するソースコード群を分析し、研究の背景、提案手法、実験設定、および期待される成果について詳細にまとめたものであり、今後の論文執筆に資することを目的とする。

### 本ドキュメントの構成
本ドキュメントは以下の構成で記述される。
- 研究背景: ヒューマノイドの歩行制御、RVC、強化学習の概要。
- 対象ロボット: Unitree G1の紹介。
- 提案手法: RVCと強化学習の統合、特に`JointRVCPositionAction`クラスの実装詳細。
- 強化学習による動作学習: MDP設計（状態、行動、報酬）、学習アルゴリズム。
- 実験設定: シミュレーション環境、学習の実行方法。
- 期待される成果と考察: 本研究によって期待される結果。
- 結論と今後の展望: 研究のまとめと将来的な方向性。
- 付録: 主要な設定値など。

## 2. 研究背景

### ヒューマノイドロボットの歩行制御の課題
ヒューマノイドロボットの二足歩行制御は、複雑な動力学、多数の自由度、環境との接触といった要因により、依然として挑戦的な課題である。従来の制御手法では、環境変化や外乱に対するロバスト性、多様な歩容の実現、計算コストなどに課題があった。

### 粘弾性分解制御（Resolved Viscoelasticity Control, RVC）
粘弾性分解制御（RVC）は、ロボットの特定部位（エンドエフェクタや重心など）に仮想的なバネ・ダンパ系（粘弾性）を想定し、その仮想的な力学系に基づいて関節の制御目標を生成する手法である。これにより、ロボットは外力に対して柔軟かつ安定した応答を示すことが期待でき、特に接触を伴う作業や不整地移動において有効性が示唆されている。本研究では、このRVCの概念を歩行制御に応用し、PD制御の目標値を生成する形で実装されている可能性がある。

### 強化学習のロボット制御への応用
強化学習（RL）は、エージェントが試行錯誤を通じて環境との相互作用から最適な行動方策を学習する枠組みである。近年、深層学習との組み合わせ（深層強化学習）により、ロボット制御の分野でも目覚ましい成果を上げており、特に複雑な動作スキルを自動で獲得させるアプローチとして注目されている。歩行制御においては、シミュレーション上で大量の試行錯誤を行うことで、多様な環境に適応可能な歩行パターンを学習させることが可能となる。

## 3. 対象ロボット: Unitree G1

### G1ヒューマノイドロボットの概要
本研究の対象ロボットは、Unitree社によって開発されたG1ヒューマノイドロボットである。G1は、比較的高度な運動能力を持つ汎用ヒューマノイドプラットフォームとして設計されている。

### シミュレーションモデル
シミュレーションには、Isaac Lab上でG1ロボットのモデルが用いられる。関連する設定ファイル（`rough_env_cfg.py`）では、`G1_CUSTOM_CFG`という設定が参照されている。
```python
# isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py
from isaaclab_assets import G1_CUSTOM_CFG
# ...
self.scene.robot = G1_CUSTOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```
`G1_CUSTOM_CFG`の具体的な物理パラメータやアクチュエータモデルの詳細は、提供されたソースコード範囲外の`isaaclab_assets`パッケージ内に定義されていると推測される。アクチュエータとしては、`ImplicitActuatorCfg`や`DCMotorCfg`などが考えられ、これらが内部的にPD制御を実行している可能性がある。

## 4. 提案手法: RVCと強化学習による歩行動作生成

本研究の核心は、RVCの概念を強化学習の行動空間の定義に組み込み、G1ヒューマノイドロボットの歩行動作を学習させる点にある。具体的には、`JointRVCPositionAction`というカスタムアクショングループを通じて実現される。

### 4.1. 粘弾性分解制御（RVC）の理論的枠組み (推定)
RVCは、ロボットのタスク空間（例：足先の位置・姿勢、胴体の姿勢）における目標インピーダンス（粘弾性）を定義し、それに基づいて関節空間での制御指令を生成する。本研究では、強化学習エージェントがRVCのパラメータ（目標位置や粘弾性特性の一部など）を出力し、`JointRVCPositionAction`がそれらを解釈して最終的な関節目標位置を計算していると推測される。

### 4.2. `JointRVCPositionAction` の実装詳細
このクラスは、`isaaclab/envs/mdp/actions/joint_actions.py` に定義されており、強化学習エージェントからのアクションを処理し、ロボットの関節への具体的な指令値を生成する。

#### 初期化処理 (`__init__`)
- 足裏の接触を検知するための接触センサー (`ContactSensor`) を左右の足首リンク（`left_ankle_roll_link`, `right_ankle_roll_link`）に設定する。これは、歩行中の支持脚と遊脚の判定に用いられる。
```python
# isaaclab/envs/mdp/actions/joint_actions.py (抜粋)
class JointRVCPositionAction(JointAction):
    def __init__(self, cfg: actions_cfg.JointRVCPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # ...
        robot_prim_path = self._asset.cfg.prim_path
        self._left_foot_sensor_cfg = ContactSensorCfg(
            prim_path=f"{robot_prim_path}/left_ankle_roll_link", # ...
        )
        self._left_foot_sensor = ContactSensor(cfg=self._left_foot_sensor_cfg)
        # ... (同様に右足も)
```

#### `apply_actions` メソッドにおける処理フロー
このメソッドがRVC制御の計算を実行する中心部分である。
1.  **ヤコビアン行列の取得**:
    *   ロボットの重心(CoM)に関するヤコビアン: `asset.get_computed_jacobian("com")`
    *   左右の足首リンクに関するヤコビアン: `asset.get_computed_jacobian("body", link_names=["left_ankle_roll_link", "right_ankle_roll_link"])`
2.  **接触状態の判定**:
    *   `_left_foot_sensor` および `_right_foot_sensor` からのデータに基づき、各環境のロボットが両足支持 (`double_support`)、左足支持 (`single_left_support`)、右足支持 (`single_right_support`)、または空中 (`no_support`) のいずれの状態にあるかを判定する。
3.  **接触状態に応じた計算**:
    *   **両足支持 (`double_support`)**:
        *   まず片足（例：左足）を拘束したと仮定して、その条件下でのヤコビアン (`single_left_jacobian_com`) を計算する。
        *   次にもう一方の足（右足）の拘束を考慮する。右足のヤコビアンに対して特異値分解（SVD: `torch.linalg.svd`）を適用し、冗長性を利用したヌルスペース射影 (`V_2`) を計算する。これにより、両足が地面に接触しているという拘束を満たしつつ、重心運動などを制御するための関節速度を計算する。
        *   `double_foot_jacobian_com = single_left_jacobian_com @ V_2` のように、拘束条件を考慮したヤコビアンが導出される。
    *   **片足支持 (`single_left_support` / `single_right_support`)**:
        *   支持脚の速度をゼロとする拘束条件のもとで、重心や遊脚の運動を制御するための関節速度を計算する。
    *   **遊脚期 (`no_support`)**:
        *   特に地面からの拘束がないため、元の重心ヤコビアン (`jacobian_com`) を用いて計算を行う。
4.  **目標関節位置の生成**:
    *   計算された目標関節速度 (`computed_velocity`) に基づき、現在の関節位置に加算するなどして、次のステップでの目標関節位置を生成する。
    *   `self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)` のように、最終的な目標関節位置をロボットに指令する。`self.processed_actions` がRVCによって計算された目標値となる。

### 4.3. PD制御との関連性 (推定)
`JointRVCPositionAction` は目標関節位置を生成する。Isaac Labのアクチュエータモデル（例：`G1_CUSTOM_CFG` 内で定義される `ImplicitActuatorCfg` や `DCMotorCfg`）は、この目標関節位置に対して内部的にPD制御（またはそれに類する低レベル制御）を実行し、必要な関節トルクを生成していると推測される。したがって、RVCはPD制御のゲインを直接調整するのではなく、PD制御の目標値を巧みに生成することで、系全体として望ましい粘弾性特性を実現していると考えられる。

## 5. 強化学習による歩行動作の学習

### 5.1. シミュレーション環境: Isaac Lab
シミュレーションはNVIDIA Isaac Lab上で行われる。

#### タスク定義: `Isaac-Velocity-Flat-G1-v0`
学習実行スクリプト `skrl.sh` で指定されているタスク名。これは、G1ロボットが平坦な地形で指定された速度で歩行することを学習するタスクであることを示唆する。
```bash
# skrl.sh
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Velocity-Flat-G1-v0 --headless
```

#### 環境設定クラス: `G1FlatEnvCfg`
このタスクは、`isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py` に定義される `G1FlatEnvCfg` クラスによって設定される。このクラスは `G1RoughEnvCfg` を継承し、地形を平坦 (`plane`) に変更するなどの設定を行っている。
```python
# isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py
@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # ...
```
ベースとなる環境設定は `isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` に定義されている。

### 5.2. マルコフ決定過程（MDP）の設計

#### 状態空間（Observations）
`velocity_env_cfg.py` の `ObservationsCfg` および、それをカスタマイズする `G1RoughEnvCfg` で定義される。具体的な観測量としては、ロボットのベースの線形速度・角速度、重力ベクトル、関節の位置・速度、前回の行動、高さスキャン（平地設定では無効化）、目標コマンドなどが含まれると推測される。

#### 行動空間（Actions）
`velocity_env_cfg.py` の `ActionsCfg` で定義される。RVC制御を用いる場合、`joint_pos` は `mdp.JointRVCPositionActionCfg` を使用するように変更されているはずである。
```python
# velocity_env_cfg.py (RVC使用時の想定)
@configclass
class ActionsCfg:
    # joint_pos = mdp.JointPositionActionCfg(...)
    joint_pos = mdp.JointRVCPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
```
この設定により、強化学習エージェントが出力するアクションが `JointRVCPositionAction` クラスによって処理される。アクションの次元はロボットの全関節数に対応する。

#### 報酬設計（Rewards）
報酬関数は `isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py` 内の `G1Rewards` クラスで詳細に定義されている。主な報酬項は以下の通り。
- `termination_penalty`: 転倒などによるエピソード終了時のペナルティ (-60.0)。
- `track_lin_vel_xy_exp`: XY平面での目標線形速度追従（重み0.5）。
- `track_ang_vel_z_exp`: Z軸周りの目標角速度追従（重み0.2）。
- `pelvis_height`: 骨盤の高さ維持（目標0.65m、重み0.2）。
- `upright_posture`: 直立姿勢の維持（骨盤と胴体リンク、重み0.3）。
- `alternating_foot_movement`: 左右の足の交互の動きの促進（重み0.3）。
- `feet_slide`: 足の滑りに対するペナルティ（重み-0.5）。
- `joint_deviation_main_joints`: 主要関節（腕、腰、脚の付け根）の関節角度が基準値から逸脱することへのペナルティ（重み-0.1）。
- `lin_vel_z_l2`: Z軸方向の線形速度（上下動）の抑制（重み-0.05）。

これらの報酬項は、`isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py` に定義された関数（例：`mdp.track_lin_vel_xy_yaw_frame_exp`, `mdp.upright_posture`など）を利用している。

### 5.3. 学習アルゴリズムと設定

#### 強化学習ライブラリ: `skrl`
本研究では、強化学習ライブラリとして `skrl` が使用されている。

#### アルゴリズム: PPO (Proximal Policy Optimization)
`isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/skrl_flat_ppo_cfg.yaml` ファイルにPPOアルゴリズムの設定が記述されている。

#### 設定ファイル: `skrl_flat_ppo_cfg.yaml` の主要パラメータ
- **モデル (`models`)**:
    - `policy` (方策ネットワーク): GaussianMixin。出力層の前に[256, 128, 128]の隠れ層を持つMLP。活性化関数はelu。
    - `value` (価値ネットワーク): DeterministicMixin。同様に[256, 128, 128]の隠れ層を持つMLP。活性化関数はelu。
- **メモリ (`memory`)**: `RandomMemory` を使用。
- **エージェント (`agent`)**: PPO
    - `rollouts`: 100 (1回の更新あたりのステップ数)
    - `learning_epochs`: 5
    - `mini_batches`: 4
    - `discount_factor` (割引率): 0.99
    - `lambda` (GAE): 0.95
    - `learning_rate`: 2.0e-04
    - `learning_rate_scheduler`: KLAdaptiveLR (KL発散に基づく適応的学習率調整)
    - `grad_norm_clip`: 1.0 (勾配クリッピング)
    - `ratio_clip`: 0.2 (PPOのクリッピングパラメータ)
    - `value_clip`: 0.2
    - `entropy_loss_scale` (エントロピー正則化係数): 0.005
    - `value_loss_scale`: 1.0
- **トレーナー (`trainer`)**: `SequentialTrainer`
    - `timesteps`: 50,000,000 (総学習ステップ数)

## 6. 実験設定（推定）

### 学習の実行コマンド
学習は、プロジェクトルートにある `skrl.sh` スクリプトを実行することで開始される。
```bash
#!/usr/bin/bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Velocity-Flat-G1-v0 --headless
```
このコマンドは、Isaac Labの環境で `train.py` スクリプトを実行し、`Isaac-Velocity-Flat-G1-v0` タスク（G1ロボットの平地歩行）の学習をヘッドレスモードで行う。

### シミュレーション条件
- **地形**: 平坦な地形 (`G1FlatEnvCfg` の設定より)。
- **ロボット**: Unitree G1 (`G1_CUSTOM_CFG` を使用)。
- **制御**: RVC (`JointRVCPositionAction`) と強化学習 (PPO) の組み合わせ。
- **環境数**: `velocity_env_cfg.py` の `MySceneCfg` で `num_envs=2048` と設定されており、多数の環境で並列に学習が行われる。

## 7. 期待される成果と考察
（本項目はソースコードから直接読み取れる情報ではないため、一般的な考察となる。）

- **高品質な歩行動作の獲得**: RVCと強化学習の組み合わせにより、G1ヒューマノイドロボットが平坦な地形で、指定された速度で安定かつ滑らかに歩行する動作を獲得することが期待される。
- **RVC導入の効果**: RVCによってもたらされるコンプライアンス制御が、歩行の安定性や外乱応答性の向上に寄与する可能性がある。また、学習の収束性や獲得される方策の汎化性能にも影響を与えるか注目される。
- **報酬設計の妥当性**: `G1Rewards` で定義された各報酬項が、望ましい歩行パターンの学習にどのように貢献したかを分析する必要がある。
- **学習効率**: PPOアルゴリズムとハイパーパラメータ設定（`skrl_flat_ppo_cfg.yaml`）の下で、どの程度の学習ステップ数で実用的な歩行動作が獲得できるか。

## 8. 結論と今後の展望
（本項目はソースコードから直接読み取れる情報ではないため、一般的な考察となる。）

### 本研究のまとめ（推定）
本研究は、G1ヒューマノイドロボットの歩行動作生成において、粘弾性分解制御（RVC）の概念を強化学習フレームワークに統合するアプローチを提案し、その有効性をIsaac Labシミュレーション環境で検証した。`JointRVCPositionAction` というカスタムアクショングループを介してRVCを実装し、PPOアルゴリズムと詳細な報酬設計によって歩行動作を学習させた。

### 考えられる今後の研究の方向性
- **不整地歩行への拡張**: 現在の平地環境から、`G1RoughEnvCfg` で定義されるような不整地環境へと拡張し、RVCのロバスト性を検証する。
- **動的な動作への展開**: 歩行だけでなく、走行、跳躍、方向転換など、よりダイナミックな動作への適用。
- **実機検証**: シミュレーションで獲得した方策を実機のG1ロボットに転移し、その性能を評価する。
- **RVCパラメータの学習**: RVCにおける粘弾性パラメータ自体を強化学習の対象とし、タスクに応じて最適化する。
- **他の制御手法との比較**: RVCを用いない純粋な強化学習や、他のモデルベース制御手法との性能比較。

## 9. 付録

### A. 主要な設定ファイルとパラメータ（抜粋）
- **環境設定**:
    - `isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
    - `isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py`
    - `isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py`
- **RVCアクション定義**:
    - `isaaclab/envs/mdp/actions/joint_actions.py` (特に `JointRVCPositionAction` クラス)
- **報酬関数定義**:
    - `isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- **強化学習エージェント設定**:
    - `isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/skrl_flat_ppo_cfg.yaml`

### B. 参考文献（関連する可能性のある一般的な文献分野）
- Resolved Momentum Control / Resolved Viscoelasticity Control
- Impedance Control for Legged Robots
- Deep Reinforcement Learning for Locomotion
- Sim-to-Real Transfer for Robotics
- Unitree G1 Humanoid Robot (Datasheets, Related publications)

