#Author: Roopesh Kumar Reddy Kaipa
#Date: 10/06/2025
import datetime
from pathlib import Path
import pandas as pd
import pytest
from unittest.mock import Mock, patch, PropertyMock
from decimal import Decimal
from tempfile import TemporaryDirectory
from app.calculator import Calculator
from app.calculator_repl import calculator_repl
from app.calculator_config import CalculatorConfig
from app.exceptions import OperationError, ValidationError
from app.history import LoggingObserver, AutoSaveObserver
from app.operations import OperationFactory

# Fixture to initialize Calculator with a temporary directory for file paths
@pytest.fixture
def calculator():
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = CalculatorConfig(base_dir=temp_path)

        # Patch properties to use the temporary directory paths
        with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
             patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file, \
             patch.object(CalculatorConfig, 'history_dir', new_callable=PropertyMock) as mock_history_dir, \
             patch.object(CalculatorConfig, 'history_file', new_callable=PropertyMock) as mock_history_file:
            
            # Set return values to use paths within the temporary directory
            mock_log_dir.return_value = temp_path / "logs"
            mock_log_file.return_value = temp_path / "logs/calculator.log"
            mock_history_dir.return_value = temp_path / "history"
            mock_history_file.return_value = temp_path / "history/calculator_history.csv"
            
            # Return an instance of Calculator with the mocked config
            yield Calculator(config=config)

# Test Calculator Initialization

def test_calculator_initialization(calculator):
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []
    assert calculator.operation_strategy is None

# Test Logging Setup

@patch('app.calculator.logging.info')
def test_logging_setup(logging_info_mock):
    with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
         patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file:
        mock_log_dir.return_value = Path('/tmp/logs')
        mock_log_file.return_value = Path('/tmp/logs/calculator.log')
        
        # Instantiate calculator to trigger logging
        calculator = Calculator(CalculatorConfig())
        logging_info_mock.assert_any_call("Calculator initialized with configuration")

# Test Adding and Removing Observers

def test_add_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    assert observer in calculator.observers

def test_remove_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    calculator.remove_observer(observer)
    assert observer not in calculator.observers

# Test Setting Operations

def test_set_operation(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    assert calculator.operation_strategy == operation

# Test Performing Operations

def test_perform_operation_addition(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    result = calculator.perform_operation(2, 3)
    assert result == Decimal('5')

def test_perform_operation_validation_error(calculator):
    calculator.set_operation(OperationFactory.create_operation('add'))
    with pytest.raises(ValidationError):
        calculator.perform_operation('invalid', 3)

def test_perform_operation_operation_error(calculator):
    with pytest.raises(OperationError, match="No operation set"):
        calculator.perform_operation(2, 3)

# Test Undo/Redo Functionality

def test_undo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    assert calculator.history == []

def test_redo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    calculator.redo()
    assert len(calculator.history) == 1

# Test History Management

@patch('app.calculator.pd.DataFrame.to_csv')
def test_save_history(mock_to_csv, calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.save_history()
    mock_to_csv.assert_called_once()

@patch('app.calculator.pd.read_csv')
@patch('app.calculator.Path.exists', return_value=True)
def test_load_history(mock_exists, mock_read_csv, calculator):
    # Mock CSV data to match the expected format in from_dict
    mock_read_csv.return_value = pd.DataFrame({
        'operation': ['Addition'],
        'operand1': ['2'],
        'operand2': ['3'],
        'result': ['5'],
        'timestamp': [datetime.datetime.now().isoformat()]
    })
    
    # Test the load_history functionality
    try:
        calculator.load_history()
        # Verify history length after loading
        assert len(calculator.history) == 1
        # Verify the loaded values
        assert calculator.history[0].operation == "Addition"
        assert calculator.history[0].operand1 == Decimal("2")
        assert calculator.history[0].operand2 == Decimal("3")
        assert calculator.history[0].result == Decimal("5")
    except OperationError:
        pytest.fail("Loading history failed due to OperationError")
        
            
# Test Clearing History

def test_clear_history(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.clear_history()
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []

# Test REPL Commands (using patches for input/output handling)

@patch('builtins.input', side_effect=['exit'])
@patch('builtins.print')
def test_calculator_repl_exit(mock_print, mock_input):
    with patch('app.calculator.Calculator.save_history') as mock_save_history:
        calculator_repl()
        mock_save_history.assert_called_once()
        mock_print.assert_any_call("History saved successfully.")
        mock_print.assert_any_call("Goodbye!")

@patch('builtins.input', side_effect=['help', 'exit'])
@patch('builtins.print')
def test_calculator_repl_help(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nAvailable commands:")

@patch('builtins.input', side_effect=['add', '2', '3', 'exit'])
@patch('builtins.print')
def test_calculator_repl_addition(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nResult: 5")


def test_save_empty_history(tmp_path, monkeypatch):
    # Ensure saving empty history writes an empty CSV (uses to_csv)
    from app.calculator_config import CalculatorConfig
    cfg = CalculatorConfig(base_dir=tmp_path)
    calc = Calculator(config=cfg)
    # Ensure history is empty
    calc.history.clear()
    with patch('app.calculator.pd.DataFrame.to_csv') as mock_to_csv:
        calc.save_history()
        assert mock_to_csv.called


def test_load_history_no_file(tmp_path):
    from app.calculator_config import CalculatorConfig
    cfg = CalculatorConfig(base_dir=tmp_path)
    # Ensure no history file exists before instantiating Calculator
    if cfg.history_file.exists():
        cfg.history_file.unlink()
    calc = Calculator(config=cfg)
    # Should not raise and history should be empty
    assert calc.history == []


def test_load_history_empty_df(monkeypatch, tmp_path):
    from app.calculator_config import CalculatorConfig
    import pandas as pd
    cfg = CalculatorConfig(base_dir=tmp_path)
    calc = Calculator(config=cfg)
    # Mock exists and read_csv to return empty DataFrame
    monkeypatch.setattr('app.calculator.Path.exists', lambda self: True)
    monkeypatch.setattr('app.calculator.pd.read_csv', lambda f: pd.DataFrame())
    # Should not raise and history should remain empty
    calc.load_history()
    assert calc.history == []


def test_repl_exit_save_warning(monkeypatch, capsys):
    # Make save_history raise during exit to test warning path
    monkeypatch.setattr('app.calculator.Calculator.save_history', lambda self: (_ for _ in ()).throw(Exception('disk full')))
    inputs = iter(['exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    try:
        calculator_repl()
    except Exception:
        # REPL may re-raise, but capture output
        pass
    out = capsys.readouterr().out
    assert 'Warning: Could not save history' in out


def test_repl_clear_save_load_success(monkeypatch, capsys):
    # Test clear, save success and load success messages
    # Patch save_history and load_history to succeed
    monkeypatch.setattr('app.calculator.Calculator.save_history', lambda self: None)
    monkeypatch.setattr('app.calculator.Calculator.load_history', lambda self: None)
    inputs = iter(['clear', 'save', 'load', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    calculator_repl()
    out = capsys.readouterr().out
    assert 'History cleared' in out
    assert 'History saved successfully' in out or 'History saved successfully.' in out
    assert 'History loaded successfully' in out


def test_repl_unexpected_operation_exception(monkeypatch, capsys):
    # Simulate an unexpected exception during perform_operation to hit 'Unexpected error'
    def fake_create(op):
        class Op:
            def __str__(self):
                return 'Op'
        return Op()

    monkeypatch.setattr('app.calculator_repl.OperationFactory.create_operation', fake_create)
    # Make Calculator.perform_operation raise a generic Exception
    monkeypatch.setattr('app.calculator.Calculator.perform_operation', lambda self, a, b: (_ for _ in ()).throw(Exception('boom')))
    inputs = iter(['add', '2', '3', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    calculator_repl()
    out = capsys.readouterr().out
    assert 'Unexpected error' in out


# Additional REPL and calculator tests merged from test_additional_coverage.py
repl_func = calculator_repl


def test_repl_cancel_first_number(monkeypatch, capsys):
    # Simulate entering 'add', then 'cancel' at first prompt, then 'exit'
    inputs = iter(['add', 'cancel', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    with patch.object(Calculator, 'save_history') as mock_save:
        repl_func()
    captured = capsys.readouterr()
    assert 'Operation cancelled' in captured.out


def test_repl_cancel_second_number(monkeypatch, capsys):
    inputs = iter(['add', '2', 'cancel', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    with patch.object(Calculator, 'save_history') as mock_save:
        repl_func()
    captured = capsys.readouterr()
    assert 'Operation cancelled' in captured.out


def test_repl_validation_error(monkeypatch, capsys):
    # Make InputValidator.validate_number raise ValidationError
    def fake_validate(val, config):
        raise ValidationError('invalid')

    inputs = iter(['add', '2', '3', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))

    # Patch the InputValidator used inside app.calculator (imported at module load time)
    monkeypatch.setattr('app.calculator.InputValidator', type('IV', (), {'validate_number': staticmethod(fake_validate)}))

    with patch.object(Calculator, 'save_history'):
        repl_func()

    captured = capsys.readouterr()
    assert 'Error:' in captured.out


def test_repl_save_load_error(monkeypatch, capsys):
    inputs = iter(['save', 'load', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))

    with patch.object(Calculator, 'save_history', side_effect=Exception('disk full')):
        with patch.object(Calculator, 'load_history', side_effect=Exception('corrupt')):
            try:
                repl_func()
            except Exception:
                pass

    captured = capsys.readouterr()
    assert 'Error saving history' in captured.out or 'Error loading history' in captured.out or 'Error:' in captured.out


def test_repl_unknown_command(monkeypatch, capsys):
    inputs = iter(['foobar', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    with patch.object(Calculator, 'save_history'):
        repl_func()
    captured = capsys.readouterr()
    assert "Unknown command" in captured.out


def test_repl_keyboard_and_eof(monkeypatch, capsys):
    def side_effect(prompt=''):
        if not hasattr(side_effect, 'count'):
            side_effect.count = 0
        side_effect.count += 1
        if side_effect.count == 1:
            raise KeyboardInterrupt()
        raise EOFError()

    monkeypatch.setattr('builtins.input', side_effect)
    with patch.object(Calculator, 'save_history'):
        repl_func()
    captured = capsys.readouterr()
    assert 'Operation cancelled' in captured.out or 'Input terminated' in captured.out


def test_perform_operation_raises_operation_error():
    class BadOp:
        def execute(self, a, b):
            raise Exception('boom')

        def __str__(self):
            return 'BadOp'
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from app.calculator_config import CalculatorConfig

    with TemporaryDirectory() as tmp:
        cfg = CalculatorConfig(base_dir=Path(tmp))
        calc = Calculator(config=cfg)
        calc.set_operation(BadOp())
        with pytest.raises(OperationError):
            calc.perform_operation(1, 2)


def _make_calc(**overrides):
    """Helper to create a fake Calculator with overridable methods."""
    from types import SimpleNamespace
    from decimal import Decimal
    from types import SimpleNamespace as NS

    def _default():
        return None

    attrs = {
        'add_observer': lambda x: None,
        'save_history': lambda: None,
        'load_history': lambda: None,
        'show_history': lambda: [],
        'clear_history': lambda: None,
        'undo': lambda: False,
        'redo': lambda: False,
        'set_operation': lambda op: None,
        'perform_operation': lambda a, b: Decimal('5'),
        'config': NS(auto_save=False),
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def test_repl_history_non_empty(monkeypatch, capsys):
    calc = _make_calc(show_history=lambda: ["Addition(2, 3) = 5"])
    monkeypatch.setattr('app.calculator_repl.Calculator', lambda: calc)
    inputs = iter(['history', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    from app.calculator import Calculator as RealCalc
    with patch.object(RealCalc, 'save_history'):
        repl_func()
    out = capsys.readouterr().out
    assert 'Calculation History' in out
    assert '1.' in out


def test_repl_undo_redo_paths(monkeypatch, capsys):
    # Undo True
    calc = _make_calc(undo=lambda: True)
    monkeypatch.setattr('app.calculator_repl.Calculator', lambda: calc)
    inputs = iter(['undo', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    with patch.object(Calculator, 'save_history'):
        repl_func()
    out = capsys.readouterr().out
    assert 'Operation undone' in out

    # Undo False
    calc = _make_calc(undo=lambda: False)
    monkeypatch.setattr('app.calculator_repl.Calculator', lambda: calc)
    inputs = iter(['undo', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    with patch.object(Calculator, 'save_history'):
        repl_func()
    out = capsys.readouterr().out
    assert 'Nothing to undo' in out

    # Redo True
    calc = _make_calc(redo=lambda: True)
    monkeypatch.setattr('app.calculator_repl.Calculator', lambda: calc)
    inputs = iter(['redo', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    with patch.object(Calculator, 'save_history'):
        repl_func()
    out = capsys.readouterr().out
    assert 'Operation redone' in out

    # Redo False
    calc = _make_calc(redo=lambda: False)
    monkeypatch.setattr('app.calculator_repl.Calculator', lambda: calc)
    inputs = iter(['redo', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    with patch.object(Calculator, 'save_history'):
        repl_func()
    out = capsys.readouterr().out
    assert 'Nothing to redo' in out


def test_repl_save_and_load_messages(monkeypatch, capsys):
    # Save raises
    calc = _make_calc(save_history=lambda: (_ for _ in ()).throw(Exception('disk full')))
    monkeypatch.setattr('app.calculator_repl.Calculator', lambda: calc)
    inputs = iter(['save', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    with patch.object(Calculator, 'save_history'):
        try:
            repl_func()
        except Exception:
            pass
    out = capsys.readouterr().out
    assert 'Error saving history' in out or 'Error:' in out

    # Load raises
    calc = _make_calc(load_history=lambda: (_ for _ in ()).throw(Exception('corrupt')))
    monkeypatch.setattr('app.calculator_repl.Calculator', lambda: calc)
    inputs = iter(['load', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    with patch.object(Calculator, 'save_history'):
        try:
            repl_func()
        except Exception:
            pass
    out = capsys.readouterr().out
    assert 'Error loading history' in out or 'Error:' in out


def test_repl_addition_decimal_normalize(monkeypatch, capsys):
    from decimal import Decimal
    calc = _make_calc(perform_operation=lambda a, b: Decimal('5.000'))
    monkeypatch.setattr('app.calculator_repl.Calculator', lambda: calc)
    inputs = iter(['add', '2', '3', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    with patch.object(Calculator, 'save_history'):
        repl_func()
    out = capsys.readouterr().out
    assert 'Result: 5' in out


def test_repl_fatal_init(monkeypatch, capsys):
    def raise_calc():
        raise Exception('boom')

    monkeypatch.setattr('app.calculator_repl.Calculator', raise_calc)
    repl_func()
    out = capsys.readouterr().out
    assert 'Fatal error' in out


def test_repl_empty_input_unknown(monkeypatch, capsys):
    # User presses enter (empty input) then exits
    inputs = iter(['', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    with patch.object(Calculator, 'save_history'):
        calculator_repl()
    out = capsys.readouterr().out
    assert "Unknown command: ''" in out or "Unknown command: ''" in out


def test_repl_incomplete_inline_args(monkeypatch, capsys):
    # User types 'add 2' which is incomplete and should be treated as unknown
    inputs = iter(['add 2', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    with patch.object(Calculator, 'save_history'):
        calculator_repl()
    out = capsys.readouterr().out
    assert "Unknown command: 'add 2'" in out


def test_repl_help_full_output(monkeypatch):
    # Verify the full help output lines are printed
    inputs = iter(['help', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    prints = []
    def fake_print(*args, **kwargs):
        prints.append(' '.join(str(a) for a in args))
    monkeypatch.setattr('builtins.print', fake_print)
    try:
        calculator_repl()
    except Exception:
        pass
    # Check several expected help lines
    assert any('Available commands:' in p for p in prints)
    assert any('add, subtract, multiply, divide, power, root' in p for p in prints)


def test_setup_logging_failure(tmp_path):
    class BadLogFile:
        def resolve(self):
            raise Exception('logfs error')

    class FakeConfig:
        def __init__(self, base_dir):
            self.base_dir = base_dir
            self.log_dir = base_dir / 'logs'
            self.log_file = BadLogFile()
            self.history_dir = base_dir / 'history'
            self.history_file = base_dir / 'history' / 'calculator_history.csv'
            self.max_history_size = 100
            self.auto_save = False
            self.precision = 10
            self.max_input_value = 1e999
            self.default_encoding = 'utf-8'

        def validate(self):
            return None

    cfg = FakeConfig(tmp_path)
    import pytest
    with pytest.raises(Exception):
        # Initializing Calculator should attempt to setup logging and re-raise
        from app.calculator import Calculator
        Calculator(config=cfg)


def test_history_max_size_prune(tmp_path):
    from app.calculator_config import CalculatorConfig
    cfg = CalculatorConfig(base_dir=tmp_path, max_history_size=1)
    calc = Calculator(config=cfg)
    calc.set_operation(OperationFactory.create_operation('add'))
    calc.perform_operation(1, 2)
    calc.perform_operation(3, 4)
    assert len(calc.history) == 1


def test_save_history_failure(monkeypatch, tmp_path):
    # Force DataFrame.to_csv to raise to hit save_history except block
    cfg = CalculatorConfig(base_dir=tmp_path)
    calc = Calculator(config=cfg)
    calc.set_operation(OperationFactory.create_operation('add'))
    calc.perform_operation(1, 2)

    import pandas as pd
    def raise_to_csv(*args, **kwargs):
        raise Exception('disk full')

    monkeypatch.setattr(pd.DataFrame, 'to_csv', raise_to_csv)
    import pytest
    with pytest.raises(Exception):
        calc.save_history()


def test_load_history_failure(monkeypatch, tmp_path):
    # Force pd.read_csv to raise to hit load_history except block
    cfg = CalculatorConfig(base_dir=tmp_path)
    calc = Calculator(config=cfg)

    import pandas as pd
    monkeypatch.setattr('app.calculator.Path.exists', lambda self: True)
    monkeypatch.setattr(pd, 'read_csv', lambda f: (_ for _ in ()).throw(Exception('corrupt')))
    import pytest
    with pytest.raises(Exception):
        calc.load_history()


def test_get_history_dataframe_and_show_history(tmp_path):
    from app.calculator_config import CalculatorConfig
    cfg = CalculatorConfig(base_dir=tmp_path)
    calc = Calculator(config=cfg)
    calc.set_operation(OperationFactory.create_operation('add'))
    calc.perform_operation(2, 3)
    df = calc.get_history_dataframe()
    assert 'operation' in df.columns
    assert len(df) == 1
    sh = calc.show_history()
    assert isinstance(sh, list) and len(sh) == 1


def test_undo_redo_empty(tmp_path):
    from app.calculator_config import CalculatorConfig
    cfg = CalculatorConfig(base_dir=tmp_path)
    calc = Calculator(config=cfg)
    assert calc.undo() is False
    assert calc.redo() is False


def test_repl_inline_args_single_input(monkeypatch, capsys):
    # Test using inline args in a single input like 'add 2 3'
    # Patch save_history to avoid writing files on exit
    monkeypatch.setattr('app.calculator.Calculator.save_history', lambda self: None)
    inputs = iter(['add 2 3', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    calculator_repl()
    out = capsys.readouterr().out
    assert 'Result: 5' in out


def test_repl_history_empty(monkeypatch, capsys):
    # history command when history is empty should print a friendly message
    calc = _make_calc(show_history=lambda: [])
    monkeypatch.setattr('app.calculator_repl.Calculator', lambda: calc)
    inputs = iter(['history', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    with patch.object(Calculator, 'save_history'):
        calculator_repl()
    out = capsys.readouterr().out
    assert 'No calculations in history' in out


def test_repl_generic_input_exception(monkeypatch, capsys):
    # If input() raises a generic Exception, REPL should catch and print it, then continue
    calls = {'n': 0}

    def side_effect(prompt=''):
        calls['n'] += 1
        if calls['n'] == 1:
            raise Exception('boom')
        return 'exit'

    monkeypatch.setattr('builtins.input', side_effect)
    with patch.object(Calculator, 'save_history'):
        calculator_repl()
    out = capsys.readouterr().out
    assert 'Error: boom' in out

