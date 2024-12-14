from typing import List

def parse_addresses(addresses) -> List[str]:
    """Parse various address input formats into a list of individual addresses
    
    Handles formats like:
    - List of addresses (e.g. ["127.0.0.1:7821", "127.0.0.1:7822"])
    - Single address string (e.g. "127.0.0.1:7821")
    - Comma-separated addresses (e.g. "127.0.0.1:7821,127.0.0.1:7822")
    - Address with port range (e.g. "127.0.0.1:7821-7824")
    Each address can optionally include a port range.
    
    Args:
        addresses: Address specification in any supported format
        
    Returns:
        List of individual address strings with port ranges expanded
    """
    # Convert to list of strings
    if isinstance(addresses, str):
        # Handle comma-separated format
        addresses = [addr.strip() for addr in addresses.split(',')]
    elif not isinstance(addresses, list):
        addresses = [str(addresses)]
        
    # Expand any port ranges
    expanded = []
    for addr in addresses:
        if ':' not in addr:
            expanded.append(addr)
            continue
            
        # Split host and port spec
        host, port_spec = addr.rsplit(':', 1)
        if '-' in port_spec:
            start, end = map(int, port_spec.split('-'))
            expanded.extend(f"{host}:{port}" for port in range(start, end + 1))
        else:
            expanded.append(addr)
            
    return expanded
