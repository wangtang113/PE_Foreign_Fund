#!/usr/bin/env python3
"""
Test runner for PE Foreign Fund firm FX measures validation.

This script runs all tests to verify that eq_w_ear_firm and related 
firm measures are correctly calculated with currency filtering.
"""

import sys
import os
import unittest
import subprocess

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def run_all_tests():
    """Run all test modules and provide a summary."""
    
    print("="*80)
    print("PE FOREIGN FUND - FIRM FX MEASURES TEST SUITE")
    print("="*80)
    print()
    
    # Test modules to run
    test_modules = [
        'test_firm_fx_measures',
        'test_actual_data_validation', 
        'test_calculate_fund_fx_measure'
    ]
    
    results = {}
    
    for module in test_modules:
        print(f"Running {module}...")
        print("-" * 60)
        
        try:
            # Run the test module
            result = subprocess.run([
                sys.executable, '-m', 'unittest', f'tests.{module}', '-v'
            ], cwd=project_root, capture_output=True, text=True)
            
            results[module] = {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            # Print output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
                
            print(f"Result: {'PASSED' if result.returncode == 0 else 'FAILED'}")
            print()
            
        except Exception as e:
            print(f"Error running {module}: {e}")
            results[module] = {'error': str(e)}
            print()
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for module, result in results.items():
        if 'error' in result:
            status = f"ERROR: {result['error']}"
            failed += 1
        elif result['returncode'] == 0:
            status = "PASSED"
            passed += 1
        else:
            status = "FAILED"
            failed += 1
        
        print(f"{module:30} {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed! Firm FX measures are correctly implemented.")
    else:
        print(f"\n⚠️  {failed} test module(s) failed. Please review the output above.")
        return 1
    
    return 0


def run_quick_validation():
    """Run a quick validation using actual data."""
    
    print("QUICK VALIDATION - Checking firm FX measures with actual data")
    print("-" * 60)
    
    try:
        import pandas as pd
        from src.utils.dataclean_utils import group_equal_mean
        
        # Load actual data
        deals = pd.read_csv('Output_data/dta_deal.csv')
        fund_fx = pd.read_csv('Output_data/fund_fx_measure.csv')
        
        # Quick check
        deals['firm_deal_match'] = deals['Firm Currency'] == deals['Deal Currency']
        
        total_deals = len(deals)
        matching_deals = deals['firm_deal_match'].sum()
        
        print(f"✓ Loaded {total_deals:,} deals")
        print(f"✓ Found {matching_deals:,} deals with firm_currency == deal_currency ({matching_deals/total_deals*100:.1f}%)")
        
        # Check firm measures exist
        firm_cols = ['forward_fx_firm', 'realized_fx_firm', 'rer_firm']
        for col in firm_cols:
            if col in fund_fx.columns:
                non_null = fund_fx[col].notna().sum()
                print(f"✓ {col}: {non_null:,} non-null values")
            else:
                print(f"✗ {col}: Column not found")
        
        print("\n✅ Quick validation completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Quick validation failed: {e}")
        return False


if __name__ == '__main__':
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        success = run_quick_validation()
        sys.exit(0 if success else 1)
    else:
        exit_code = run_all_tests()
        sys.exit(exit_code)
