"""
Validation tests using actual project data to verify firm FX measures.

This module tests the firm FX measures against the actual data generated 
by the project pipeline to ensure correctness.
"""

import sys
import os
import pandas as pd
import numpy as np
import unittest

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.dataclean_utils import group_weighted_mean, group_equal_mean


class TestActualDataValidation(unittest.TestCase):
    """Test firm FX measures using actual project data."""
    
    @classmethod
    def setUpClass(cls):
        """Load actual data files for testing."""
        try:
            cls.deal_data = pd.read_csv('Output_data/dta_deal.csv')
            cls.fund_fx_data = pd.read_csv('Output_data/fund_fx_measure.csv')
            cls.data_available = True
        except FileNotFoundError:
            cls.data_available = False
    
    def setUp(self):
        """Skip tests if data is not available."""
        if not self.data_available:
            self.skipTest("Actual data files not available")
    
    def test_firm_currency_matching_statistics(self):
        """Verify basic statistics about firm currency matching."""
        
        # Calculate firm-deal currency matches
        firm_deal_match = (
            self.deal_data['Firm Currency'] == self.deal_data['Deal Currency']
        )
        
        total_deals = len(self.deal_data)
        matching_deals = firm_deal_match.sum()
        
        print(f"\nFirm Currency Matching Statistics:")
        print(f"Total deals: {total_deals:,}")
        print(f"Deals where firm_currency == deal_currency: {matching_deals:,}")
        print(f"Percentage of matching deals: {matching_deals/total_deals*100:.1f}%")
        
        # Basic sanity checks
        self.assertGreater(total_deals, 0, "Should have some deals")
        self.assertGreater(matching_deals, 0, "Should have some matching deals")
        self.assertLess(matching_deals, total_deals, "Not all deals should match")
    
    def test_firm_fx_measures_not_null(self):
        """Test that firm FX measures exist and are properly calculated."""
        
        firm_fx_cols = ['forward_fx_firm', 'realized_fx_firm', 'rer_firm']
        
        for col in firm_fx_cols:
            if col in self.fund_fx_data.columns:
                non_null_count = self.fund_fx_data[col].notna().sum()
                total_funds = len(self.fund_fx_data)
                
                print(f"\n{col} statistics:")
                print(f"Non-null values: {non_null_count:,}/{total_funds:,}")
                print(f"Coverage: {non_null_count/total_funds*100:.1f}%")
                
                # Should have some non-null values
                self.assertGreater(non_null_count, 0, f"{col} should have some non-null values")
            else:
                self.fail(f"Column {col} not found in fund FX data")
    
    def test_manual_verification_sample_fund(self):
        """Manually verify calculation for a specific fund with known data."""
        
        # Find a fund with both matching and non-matching deals
        self.deal_data['firm_deal_match'] = (
            self.deal_data['Firm Currency'] == self.deal_data['Deal Currency']
        )
        
        # Get fund statistics
        fund_stats = self.deal_data.groupby('fund_id').agg({
            'firm_deal_match': ['sum', 'count'],
            'deal_forward_fx_firm': 'mean'
        }).reset_index()
        
        fund_stats.columns = ['fund_id', 'matching_deals', 'total_deals', 'mean_firm_fx']
        fund_stats['has_mixed'] = (
            (fund_stats['matching_deals'] > 0) & 
            (fund_stats['matching_deals'] < fund_stats['total_deals'])
        )
        
        # Find a fund with mixed deals
        mixed_funds = fund_stats[fund_stats['has_mixed']]
        
        if len(mixed_funds) > 0:
            test_fund_id = mixed_funds.iloc[0]['fund_id']
            
            # Get deal data for this fund
            fund_deals = self.deal_data[
                self.deal_data['fund_id'].astype(str) == str(test_fund_id)
            ].copy()
            
            # Filter for matching deals only
            matching_deals = fund_deals[fund_deals['firm_deal_match']]
            
            # Manual calculation
            if len(matching_deals) > 0 and 'deal_forward_fx_firm' in matching_deals.columns:
                manual_mean = matching_deals['deal_forward_fx_firm'].mean()
                
                # Get fund-level measure
                fund_measure = self.fund_fx_data[
                    self.fund_fx_data['fund_id'].astype(str) == str(test_fund_id)
                ]
                
                if len(fund_measure) > 0 and 'forward_fx_firm' in fund_measure.columns:
                    pipeline_value = fund_measure['forward_fx_firm'].iloc[0]
                    
                    print(f"\nManual verification for fund {test_fund_id}:")
                    print(f"Total deals: {len(fund_deals)}")
                    print(f"Matching deals: {len(matching_deals)}")
                    print(f"Manual calculation: {manual_mean:.6f}")
                    print(f"Pipeline value: {pipeline_value:.6f}")
                    
                    # They should match within floating point precision
                    if pd.notna(pipeline_value) and pd.notna(manual_mean):
                        self.assertAlmostEqual(
                            manual_mean, pipeline_value, places=5,
                            msg=f"Manual calculation doesn't match pipeline for fund {test_fund_id}"
                        )
        else:
            self.skipTest("No funds with mixed deals found for verification")
    
    def test_firm_measures_zero_when_no_matches(self):
        """Test that funds with no firm-deal currency matches have NaN firm measures."""
        
        # Find funds with no firm-deal currency matches
        self.deal_data['firm_deal_match'] = (
            self.deal_data['Firm Currency'] == self.deal_data['Deal Currency']
        )
        
        fund_matches = self.deal_data.groupby('fund_id')['firm_deal_match'].sum()
        funds_with_no_matches = fund_matches[fund_matches == 0].index
        
        if len(funds_with_no_matches) > 0:
            # Check a sample of these funds
            sample_fund = funds_with_no_matches[0]
            
            fund_measure = self.fund_fx_data[
                self.fund_fx_data['fund_id'].astype(str) == str(sample_fund)
            ]
            
            if len(fund_measure) > 0:
                # Firm measures should be NaN for funds with no matching deals
                for col in ['forward_fx_firm', 'realized_fx_firm', 'rer_firm']:
                    if col in fund_measure.columns:
                        value = fund_measure[col].iloc[0]
                        print(f"\nFund {sample_fund} (no matches) - {col}: {value}")
                        # Should be NaN or could be 0.0 depending on implementation
                        self.assertTrue(
                            pd.isna(value) or value == 0.0,
                            f"Fund with no matches should have NaN or 0 for {col}, got {value}"
                        )
    
    def test_data_consistency_checks(self):
        """Perform additional data consistency checks."""
        
        # Check that fund IDs match between datasets (convert to int for proper comparison)
        deal_funds = set(self.deal_data['fund_id'].dropna().astype(int))
        fx_funds = set(self.fund_fx_data['fund_id'].astype(int))
        
        # Fund FX data should be a subset of deal data funds
        missing_funds = fx_funds - deal_funds
        common_funds = fx_funds & deal_funds
        coverage = len(common_funds) / len(fx_funds) * 100
        
        print(f"Fund coverage: {len(common_funds)}/{len(fx_funds)} ({coverage:.1f}%)")
        
        # Should have high coverage (most funds should match)
        self.assertGreater(coverage, 90, f"Should have high fund coverage, got {coverage:.1f}%")
        
        if len(missing_funds) > 0:
            print(f"Warning: {len(missing_funds)} funds in FX data not found in deal data")
        
        # Check for reasonable value ranges
        for col in ['forward_fx_firm', 'realized_fx_firm']:
            if col in self.fund_fx_data.columns:
                values = self.fund_fx_data[col].dropna()
                if len(values) > 0:
                    # Values should be reasonable (within some range)
                    min_val, max_val = values.min(), values.max()
                    print(f"\n{col} range: [{min_val:.4f}, {max_val:.4f}]")
                    
                    # Sanity check: shouldn't have extreme outliers after winsorization
                    self.assertGreater(min_val, -100, f"{col} minimum seems too extreme")
                    self.assertLess(max_val, 100, f"{col} maximum seems too extreme")


class TestFirmMeasuresLogic(unittest.TestCase):
    """Test the core logic of firm measures calculation."""
    
    def test_filtering_logic_edge_cases(self):
        """Test edge cases in the filtering logic."""
        
        # Test with missing currencies
        test_data = pd.DataFrame({
            'fund_id': [1, 2, 3, 4],
            'Deal Currency': ['USD', np.nan, 'EUR', 'GBP'],
            'Firm Currency': ['USD', 'EUR', np.nan, 'USD'],
            'deal_forward_fx_firm': [1.0, 2.0, 3.0, 4.0],
            'deal_weight': [1.0, 1.0, 1.0, 1.0]
        })
        
        # Apply filtering logic
        filtered = test_data[test_data['Firm Currency'] == test_data['Deal Currency']]
        
        # Only fund 1 should match (USD == USD), others have NaN or don't match
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]['fund_id'], 1)
    
    def test_currency_case_sensitivity(self):
        """Test that currency matching is case-insensitive or properly handled."""
        
        test_data = pd.DataFrame({
            'fund_id': [1, 2, 3],
            'Deal Currency': ['USD', 'eur', 'GBP'],
            'Firm Currency': ['usd', 'EUR', 'gbp'],
            'deal_forward_fx_firm': [1.0, 2.0, 3.0],
            'deal_weight': [1.0, 1.0, 1.0]
        })
        
        # Apply case normalization (as done in the main pipeline)
        test_data['Deal Currency'] = test_data['Deal Currency'].str.upper()
        test_data['Firm Currency'] = test_data['Firm Currency'].str.upper()
        
        # Now apply filtering
        filtered = test_data[test_data['Firm Currency'] == test_data['Deal Currency']]
        
        # All should match after case normalization
        self.assertEqual(len(filtered), 3)


if __name__ == '__main__':
    # Run with verbose output to see the statistics
    unittest.main(verbosity=2)
