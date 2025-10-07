#Author: Roopesh Kumar Reddy Kaipa
#Date: 10/06/2025
import datetime
from app.calculator_memento import CalculatorMemento
from app.calculation import Calculation
from decimal import Decimal


def test_memento_to_from_dict():
    calc = Calculation(operation="Addition", operand1=Decimal("1"), operand2=Decimal("2"))
    m = CalculatorMemento(history=[calc])
    d = m.to_dict()

    m2 = CalculatorMemento.from_dict(d)
    assert len(m2.history) == 1
    assert m2.history[0].operation == "Addition"
    assert isinstance(m2.timestamp, datetime.datetime)
