### Setup

Add the following environment variable to `~/.bashrc`:
```
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```
and `source ~/.bashrc`.

Create repo and download code:
```
mkdir ~/code
cd ~/code
git clone https://github.com/pauldb89/games.git
```

Create and start virtualenv:
```
mkdir ~/.virtualenvs
python -m venv ~/.virtualenvs/games
source ~/.virtualenvs/games/bin/activate
cd ~/code/games
pip install -r requirements.in
```

Run tests:
```
cd ~/code/games
PYTHONPATH=. pytest tests/wordle
```

Launch trainer:
```
PYTHONPATH=. torchrun --nproc_per_node=8 --nnodes=1 wordle/scripts/train.py --vocab_path /root/code/games/wordle/vocab.txt --bootstrap_values --name vocabfull-allsecrets-ep10k-8x --num_episodes_per_epoch 10000 --evaluate_every_n_epochs 100
```