import pytest
from src.data_processing import clean_column_names

def test_clean_column_names_spaces():
    assert clean_column_names("Credit Score ") == "credit_score"

def test_clean_column_names_special_chars():
    assert clean_column_names("Customer-Income$") == "customer_income"
