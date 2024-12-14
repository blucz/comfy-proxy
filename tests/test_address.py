import pytest
from comfy_proxy.address import parse_addresses

def test_single_address() -> None:
    assert parse_addresses("127.0.0.1:7821") == ["127.0.0.1:7821"]
    
def test_list_of_addresses() -> None:
    input_list = ["127.0.0.1:7821", "127.0.0.1:7822"]
    assert parse_addresses(input_list) == input_list
    
def test_comma_separated() -> None:
    assert parse_addresses("127.0.0.1:7821,127.0.0.1:7822") == [
        "127.0.0.1:7821",
        "127.0.0.1:7822"
    ]
    
def test_port_range() -> None:
    assert parse_addresses("127.0.0.1:7821-7823") == [
        "127.0.0.1:7821",
        "127.0.0.1:7822", 
        "127.0.0.1:7823"
    ]
    
def test_mixed_formats() -> None:
    assert parse_addresses("127.0.0.1:7821-7822,127.0.0.1:7824") == [
        "127.0.0.1:7821",
        "127.0.0.1:7822",
        "127.0.0.1:7824"
    ]
    
def test_whitespace_handling() -> None:
    assert parse_addresses(" 127.0.0.1:7821 , 127.0.0.1:7822 ") == [
        "127.0.0.1:7821",
        "127.0.0.1:7822"
    ]
    
def test_non_string_input() -> None:
    assert parse_addresses(7821) == ["7821"]
    
def test_no_port() -> None:
    assert parse_addresses("localhost") == ["localhost"]
    
def test_multiple_port_ranges() -> None:
    assert parse_addresses([
        "127.0.0.1:7821-7822",
        "127.0.0.1:7824-7825"
    ]) == [
        "127.0.0.1:7821",
        "127.0.0.1:7822",
        "127.0.0.1:7824",
        "127.0.0.1:7825"
    ]
