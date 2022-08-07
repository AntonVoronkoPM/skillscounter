### Virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e ".[dev]"
pre-commit install
pre-commit autoupdate
```

### Directory
```bash
skillscounter/
├── data.py       - data processing utilities
├── main.py       - training/optimization operations
├── train.py      - training utilities
└── utils.py      - supplementary utilities
```

### Workflow
```bash
python skillscounter/main.py load-data
python skillscounter/main.py optimize --args-fp="config/args.json" --study-name="optimization" --num-trials=10
python skillscounter/main.py train-model --args-fp="config/args.json" --experiment-name="baselines" --run-name="voter"
python skillscounter/main.py predict --text=["Project management"]
```