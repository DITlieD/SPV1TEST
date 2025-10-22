#!/usr/bin/env python3
"""
Quick script to check available symbols on Bybit testnet
"""
import ccxt.pro as ccxt
import asyncio

async def check_bybit_symbols():
    print("Checking Bybit Testnet symbols...")

    # Initialize Bybit testnet
    exchange = ccxt.bybit({
        'options': {
            'defaultType': 'swap',
        },
    })
    exchange.set_sandbox_mode(True)

    try:
        # Load markets
        markets = await exchange.load_markets()

        # Filter for USDT perpetuals
        usdt_perps = [symbol for symbol in markets.keys() if 'USDT' in symbol and ':USDT' in symbol]

        print(f"\nFound {len(usdt_perps)} USDT perpetual contracts on Bybit Testnet:")
        print("-" * 50)

        # Check for our specific symbols
        target_symbols = ['EPIC/USDT:USDT', 'ARC/USDT:USDT', 'DOGE/USDT:USDT', 'BTC/USDT:USDT']

        for target in target_symbols:
            status = "[AVAILABLE]" if target in usdt_perps else "[NOT FOUND]"
            print(f"{target:20s} {status}")

        # Check if there are similar symbols
        print("\n" + "=" * 50)
        print("Searching for EPIC-like symbols:")
        epic_like = [s for s in markets.keys() if 'EPIC' in s.upper()]
        if epic_like:
            for sym in epic_like:
                print(f"  - {sym}")
        else:
            print("  No EPIC symbols found")

        print("\nSearching for ARC-like symbols:")
        arc_like = [s for s in markets.keys() if 'ARC' in s.upper()]
        if arc_like:
            for sym in arc_like:
                print(f"  - {sym}")
        else:
            print("  No ARC symbols found")

        print("\n" + "=" * 50)
        print(f"\nAll available USDT perpetuals (showing first 20):")
        for symbol in sorted(usdt_perps)[:20]:
            print(f"  {symbol}")

        if len(usdt_perps) > 20:
            print(f"  ... and {len(usdt_perps) - 20} more")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(check_bybit_symbols())
