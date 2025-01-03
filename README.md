### **Lux AI Season 3 RL Kit: README**

Welcome to the **Lux AI Season 3 RL Kit**! This repository is designed to help you develop, train, and evaluate an AI agent for the Lux AI Season 3 competition using reinforcement learning.

---

## **Getting Started**

### **1. Requirements**
- **Python**: Version 3.10 or higher
- **Libraries**:
  - `luxai-s3`
  - `stable-baselines3`
  - `numpy`

Install the dependencies:
```bash
pip install --upgrade luxai-s3 stable-baselines3 numpy
```

### **2. Training the agent**

To train the reinforcement learning (RL) agent, run the following command:
```bash
python train.py --n-envs 10 --log-path logs/exp_1 --seed 42
```

- **This Wills**:
    - `Train the agent using the PPO algorithm with 10 parallel environments.`
    - `Save the model and training logs to logs/exp_1.`

### **3. Monitoring Training**
You can monitor training progress using TensorBoard:

```bash
tensorboard --logdir logs
```

### **4. Getting Started**
To verify your installation, you can run a match between two random agents:

```bash
luxai-s3 --help
```

```bash
luxai-s3 path/to/bot/main.py path/to/bot/main.py --output replay.json
```

Then upload the replay.json to the online visualizer here: https://s3vis.lux-ai.org/ (a link on the lux-ai.org website will be up soon) 

### **5. Submitting to Kaggle**

- **To create a submission package:**:
    - `Ensure your trained model is saved in logs/exp_1/models/best_model.zip.`
    - `Update MODEL_WEIGHTS_RELATIVE_PATH in agent.py if necessary.`
    
Bundle your files:
```bash
 tar -czvf submission.tar.gz *
```

Upload submission.tar.gz to the Kaggle competition page.

