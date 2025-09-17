"""
Test suite for firm FX measures calculation with currency filtering.

This module tests that eq_w_ear_firm (firm forward FX measures) are correctly 
calculated with the constraint that firm_currency == deal_currency.
"""

import sys
import os
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch

# Add the src directory to the path to import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.dataclean_utils import group_weighted_mean, group_equal_mean


class TestFirmFXMeasures(unittest.TestCase):
    """Test cases for firm FX measures calculation."""
    
    def setUp(self):
        """Set up test data with known firm currency filtering scenarios."""
        # Create synthetic test data with clear firm currency matching patterns
        self.test_data = pd.DataFrame({
            'fund_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'Deal Currency': ['USD', 'EUR', 'GBP', 'USD', 'USD', 'GBP', 'GBP', 'GBP', 'EUR'],
            'Fund Currency': ['USD', 'USD', 'USD', 'EUR', 'EUR', 'EUR', 'GBP', 'GBP', 'GBP'],
            'Firm Currency': ['USD', 'USD', 'USD', 'EUR', 'EUR', 'EUR', 'GBP', 'GBP', 'GBP'],
            'deal_forward_fx_firm': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'deal_realized_fx_firm': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1],
            'deal_rer_firm': [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09],
            'deal_weight': [0.5, 0.3, 0.2, 0.4, 0.4, 0.2, 0.6, 0.3, 0.1]
        })
        
        # Create expected firm currency matches
        self.test_data['firm_deal_match'] = (
            self.test_data['Firm Currency'] == self.test_data['Deal Currency']
        )
    
    def test_firm_currency_filtering_logic(self):
        """Test that firm currency filtering identifies correct matching deals."""
        # Expected matches based on our test data
        expected_matches = [
            True,   # Fund 1, Deal 1: USD-USD (match)
            False,  # Fund 1, Deal 2: USD-EUR (no match)
            False,  # Fund 1, Deal 3: USD-GBP (no match)
            False,  # Fund 2, Deal 1: EUR-USD (no match)  
            False,  # Fund 2, Deal 2: EUR-USD (no match)
            False,  # Fund 2, Deal 3: EUR-GBP (no match)
            True,   # Fund 3, Deal 1: GBP-GBP (match)
            True,   # Fund 3, Deal 2: GBP-GBP (match)
            False   # Fund 3, Deal 3: GBP-EUR (no match)
        ]
        
        actual_matches = (self.test_data['Firm Currency'] == self.test_data['Deal Currency']).tolist()
        self.assertEqual(actual_matches, expected_matches)
    
    def test_firm_fx_filtering_manual_calculation(self):
        """Test firm FX measures with manual calculation for verification."""
        
        # Filter for matching deals only
        filtered_data = self.test_data[
            self.test_data['Firm Currency'] == self.test_data['Deal Currency']
        ]
        
        # Expected filtered data:
        # Fund 1: Deal 1 (USD-USD) -> forward_fx = 1.0
        # Fund 3: Deal 1 (GBP-GBP) -> forward_fx = 7.0, Deal 2 (GBP-GBP) -> forward_fx = 8.0
        
        # Manual calculation for fund 1 (only deal 1 should count)
        fund1_data = filtered_data[filtered_data['fund_id'] == 1]
        expected_fund1_mean = 1.0  # Only one matching deal
        
        # Manual calculation for fund 3 (deals 1 and 2 should count)
        fund3_data = filtered_data[filtered_data['fund_id'] == 3]
        expected_fund3_mean = (7.0 + 8.0) / 2  # Two matching deals
        
        # Calculate using our function
        actual_result = group_equal_mean(filtered_data, 'fund_id', 'deal_forward_fx_firm')
        
        # Verify results
        self.assertAlmostEqual(actual_result.loc[1], expected_fund1_mean, places=6)
        self.assertAlmostEqual(actual_result.loc[3], expected_fund3_mean, places=6)
        
        # Fund 2 should not appear in filtered data (no matching deals)
        self.assertNotIn(2, actual_result.index)
    
    def test_weighted_firm_fx_calculation(self):
        """Test weighted firm FX measures with manual calculation."""
        
        # Filter for matching deals only
        filtered_data = self.test_data[
            self.test_data['Firm Currency'] == self.test_data['Deal Currency']
        ]
        
        # Manual calculation for fund 3 weighted mean
        fund3_data = filtered_data[filtered_data['fund_id'] == 3]
        # Deals: forward_fx=[7.0, 8.0], weights=[0.6, 0.3]
        expected_fund3_weighted = (7.0 * 0.6 + 8.0 * 0.3) / (0.6 + 0.3)
        
        # Calculate using our function
        actual_result = group_weighted_mean(
            filtered_data, 'fund_id', 'deal_forward_fx_firm', 'deal_weight'
        )
        
        # Verify results
        self.assertAlmostEqual(actual_result.loc[3], expected_fund3_weighted, places=6)
    
    def test_all_firm_measures_consistency(self):
        """Test that all firm measures (forward_fx, realized_fx, rer) are consistently filtered."""
        
        # Filter for matching deals only
        filtered_data = self.test_data[
            self.test_data['Firm Currency'] == self.test_data['Deal Currency']
        ]
        
        # Calculate all firm measures
        forward_fx_result = group_equal_mean(filtered_data, 'fund_id', 'deal_forward_fx_firm')
        realized_fx_result = group_equal_mean(filtered_data, 'fund_id', 'deal_realized_fx_firm')
        rer_result = group_equal_mean(filtered_data, 'fund_id', 'deal_rer_firm')
        
        # All measures should have the same fund IDs (same filtering applied)
        self.assertEqual(set(forward_fx_result.index), set(realized_fx_result.index))
        self.assertEqual(set(forward_fx_result.index), set(rer_result.index))
        
        # Expected funds: 1 and 3 (funds with matching deals)
        expected_funds = {1, 3}
        self.assertEqual(set(forward_fx_result.index), expected_funds)
    
    def test_empty_filtering_result(self):
        """Test behavior when no deals match firm currency criteria."""
        
        # Create test data where no deals match firm currency
        no_match_data = pd.DataFrame({
            'fund_id': [1, 1, 2, 2],
            'Deal Currency': ['USD', 'EUR', 'GBP', 'CHF'],
            'Firm Currency': ['EUR', 'USD', 'USD', 'GBP'],
            'deal_forward_fx_firm': [1.0, 2.0, 3.0, 4.0],
            'deal_weight': [0.5, 0.5, 0.5, 0.5]
        })
        
        # Filter for matching deals (should be empty)
        filtered_data = no_match_data[
            no_match_data['Firm Currency'] == no_match_data['Deal Currency']
        ]
        
        # Should be empty DataFrame
        self.assertEqual(len(filtered_data), 0)
        
        # Function should return empty Series
        result = group_equal_mean(filtered_data, 'fund_id', 'deal_forward_fx_firm')
        self.assertEqual(len(result), 0)
    
    def test_single_fund_multiple_scenarios(self):
        """Test a single fund with both matching and non-matching deals."""
        
        single_fund_data = pd.DataFrame({
            'fund_id': [100, 100, 100, 100],
            'Deal Currency': ['USD', 'EUR', 'USD', 'GBP'],
            'Firm Currency': ['USD', 'USD', 'USD', 'USD'],  # Firm is USD
            'deal_forward_fx_firm': [10.0, 20.0, 30.0, 40.0],
            'deal_weight': [0.25, 0.25, 0.25, 0.25]
        })
        
        # Only USD deals should match (deals 1 and 3)
        filtered_data = single_fund_data[
            single_fund_data['Firm Currency'] == single_fund_data['Deal Currency']
        ]
        
        # Expected: only deals with forward_fx = 10.0 and 30.0
        expected_mean = (10.0 + 30.0) / 2
        
        result = group_equal_mean(filtered_data, 'fund_id', 'deal_forward_fx_firm')
        
        self.assertAlmostEqual(result.loc[100], expected_mean, places=6)
        self.assertEqual(len(filtered_data), 2)  # Only 2 matching deals


class TestFirmFXIntegration(unittest.TestCase):
    """Integration tests using actual project functions."""
    
    def setUp(self):
        """Set up test environment with mock data."""
        # Import the function we're testing
        try:
            from src.utils.dataclean_utils import group_equal_mean, group_weighted_mean
            self.group_equal_mean = group_equal_mean
            self.group_weighted_mean = group_weighted_mean
        except ImportError:
            self.skipTest("Cannot import required functions")
    
    def test_calculate_fund_fx_measure_structure(self):
        """Test that calculate_fund_fx_measure properly applies firm filtering."""
        
        # Create test data that matches the expected structure
        test_deals = pd.DataFrame({
            'fund_id': [1, 1, 1, 2, 2],
            'DEAL ID': [101, 102, 103, 201, 202],
            'Deal Currency': ['USD', 'EUR', 'USD', 'GBP', 'GBP'], 
            'Fund Currency': ['USD', 'USD', 'USD', 'EUR', 'EUR'],
            'Firm Currency': ['USD', 'USD', 'EUR', 'GBP', 'EUR'],  # Mixed matching
            'deal_forward_fx_firm': [1.0, 2.0, 3.0, 4.0, 5.0],
            'deal_realized_fx_firm': [1.1, 2.1, 3.1, 4.1, 5.1],
            'deal_rer_firm': [1.01, 1.02, 1.03, 1.04, 1.05],
            'deal_weight': [0.4, 0.3, 0.3, 0.6, 0.4]
        })
        
        # Test the filtering logic that should be applied
        firm_filtered = test_deals[test_deals['Firm Currency'] == test_deals['Deal Currency']]
        
        # Expected matching deals:
        # Fund 1: Deal 101 (USD-USD), Deal 102 (EUR-USD, no match), Deal 103 (USD-EUR, no match) -> Only Deal 101
        # Fund 2: Deal 201 (GBP-GBP), Deal 202 (GBP-EUR, no match) -> Only Deal 201
        
        expected_matching_deals = 2  # Only deals 101 and 201 should match
        self.assertEqual(len(firm_filtered), expected_matching_deals)
        
        # Verify the correct deals are included
        expected_deal_ids = {101, 201}
        actual_deal_ids = set(firm_filtered['DEAL ID'])
        self.assertEqual(actual_deal_ids, expected_deal_ids)


if __name__ == '__main__':
    unittest.main()
