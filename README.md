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
TODO: fill me.
```