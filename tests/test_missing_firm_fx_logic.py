"""
Test suite to verify the logic behind missing firm FX measures.

This module tests the hypothesis that missing forward_fx_firm values indicate
that a fund never invested in deals denominated in the firm's currency.
"""

import sys
import os
import pandas as pd
import numpy as np
import unittest

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestMissingFirmFXLogic(unittest.TestCase):
    """Test cases for missing firm FX measures logic."""
    
    @classmethod
    def setUpClass(cls):
        """Load data files for testing."""
        try:
            cls.dta_fund = pd.read_csv('Output_data/dta_fund.csv')
            cls.dta_deal = pd.read_csv('Output_data/dta_deal.csv')
            cls.fund_fx = pd.read_csv('Output_data/fund_fx_measure.csv')
            cls.data_available = True
        except FileNotFoundError as e:
            print(f"Data file not found: {e}")
            cls.data_available = False
    
    def setUp(self):
        """Skip tests if data is not available."""
        if not self.data_available:
            self.skipTest("Required data files not available")
    
    def test_fund_3971_specific_case(self):
        """Test the specific case of fund 3971 mentioned by the user."""
        
        # Get fund 3971 details
        fund_3971 = self.dta_fund[self.dta_fund['fund_id'] == 3971.0]
        
        # Verify fund exists
        self.assertEqual(len(fund_3971), 1, "Fund 3971 should exist in the data")
        
        # Verify fund characteristics
        firm_currency = fund_3971['firm_currency'].iloc[0]
        fund_currency = fund_3971['fund_currency'].iloc[0]
        
        self.assertEqual(firm_currency, 'EUR', "Fund 3971 should have EUR firm currency")
        self.assertEqual(fund_currency, 'USD', "Fund 3971 should have USD fund currency")
        
        # Check forward_fx_firm value
        if 'forward_fx_firm' in fund_3971.columns:
            forward_fx_firm = fund_3971['forward_fx_firm'].iloc[0]
            self.assertTrue(pd.isna(forward_fx_firm), "Fund 3971 should have missing forward_fx_firm")
        
        # Check deals for fund 3971
        deals_3971 = self.dta_deal[self.dta_deal['fund_id'] == 3971.0]
        
        self.assertGreater(len(deals_3971), 0, "Fund 3971 should have some deals")
        
        # Check if any deals are in EUR (firm currency)
        eur_deals = deals_3971[deals_3971['Deal Currency'] == 'EUR']
        self.assertEqual(len(eur_deals), 0, "Fund 3971 should have no EUR-denominated deals")
        
        # Verify the matching logic
        deals_3971_copy = deals_3971.copy()
        deals_3971_copy['firm_deal_match'] = (
            deals_3971_copy['Firm Currency'] == deals_3971_copy['Deal Currency']
        )
        matching_deals = deals_3971_copy['firm_deal_match'].sum()
        
        self.assertEqual(matching_deals, 0, "Fund 3971 should have no deals matching firm currency")
        
        print(f"\n✓ Fund 3971 validation:")
        print(f"  Firm currency: {firm_currency}")
        print(f"  Fund currency: {fund_currency}")
        print(f"  Total deals: {len(deals_3971)}")
        print(f"  EUR deals: {len(eur_deals)}")
        print(f"  Matching deals: {matching_deals}")
        print(f"  Deal currencies: {deals_3971['Deal Currency'].value_counts().to_dict()}")
    
    def test_missing_firm_fx_hypothesis_systematic(self):
        """Systematically test the hypothesis that missing firm FX = no matching currency deals."""
        
        # Get funds with missing forward_fx_firm from fund_fx data
        funds_with_missing_firm_fx = self.fund_fx[
            self.fund_fx['forward_fx_firm'].isna()
        ]['fund_id'].tolist()
        
        print(f"\nFound {len(funds_with_missing_firm_fx)} funds with missing forward_fx_firm")
        
        if len(funds_with_missing_firm_fx) == 0:
            self.skipTest("No funds with missing forward_fx_firm found")
        
        hypothesis_confirmed = 0
        hypothesis_violated = 0
        
        # Test ALL funds with missing forward_fx_firm for comprehensive validation
        for fund_id in funds_with_missing_firm_fx:
            # Get fund details (use direct comparison, not string conversion)
            fund_data = self.dta_fund[self.dta_fund['fund_id'] == fund_id]
            
            if len(fund_data) == 0:
                continue  # Skip if fund not found in dta_fund
            
            firm_currency = fund_data['firm_currency'].iloc[0]
            
            # Get deals for this fund (use direct comparison)
            deals = self.dta_deal[self.dta_deal['fund_id'] == fund_id]
            
            if len(deals) == 0:
                continue  # Skip if no deals found
            
            # Check for deals in firm currency
            firm_currency_deals = deals[deals['Deal Currency'] == firm_currency]
            
            # Test hypothesis
            if len(firm_currency_deals) == 0:
                hypothesis_confirmed += 1
            else:
                hypothesis_violated += 1
                print(f"  VIOLATION: Fund {fund_id} (firm: {firm_currency}) has {len(firm_currency_deals)} deals in firm currency")
        
        total_tested = hypothesis_confirmed + hypothesis_violated
        accuracy = hypothesis_confirmed / total_tested * 100 if total_tested > 0 else 0
        
        print(f"  Tested {total_tested} funds")
        print(f"  Hypothesis confirmed: {hypothesis_confirmed}")
        print(f"  Hypothesis violated: {hypothesis_violated}")
        print(f"  Accuracy: {accuracy:.1f}%")
        
        # The hypothesis should be correct in all or nearly all cases
        self.assertGreaterEqual(accuracy, 95, f"Hypothesis should be correct in 95%+ of cases, got {accuracy:.1f}%")
    
    def test_non_missing_firm_fx_has_matching_deals(self):
        """Test the converse: funds with non-missing firm FX should have matching deals."""
        
        # Get funds with non-missing forward_fx_firm
        funds_with_firm_fx = self.fund_fx[
            self.fund_fx['forward_fx_firm'].notna()
        ]['fund_id'].tolist()
        
        print(f"\nTesting sample of funds with non-missing forward_fx_firm")
        
        if len(funds_with_firm_fx) == 0:
            self.skipTest("No funds with non-missing forward_fx_firm found")
        
        # Test a random sample for efficiency
        import random
        random.seed(42)  # For reproducibility
        sample_size = min(50, len(funds_with_firm_fx))
        sample_funds = random.sample(funds_with_firm_fx, sample_size)
        
        has_firm_deals = 0
        no_firm_deals = 0
        
        for fund_id in sample_funds:
            # Get fund details (use direct comparison)
            fund_data = self.dta_fund[self.dta_fund['fund_id'] == fund_id]
            
            if len(fund_data) == 0:
                continue
            
            firm_currency = fund_data['firm_currency'].iloc[0]
            
            # Get deals for this fund (use direct comparison)
            deals = self.dta_deal[self.dta_deal['fund_id'] == fund_id]
            
            if len(deals) == 0:
                continue
            
            # Check for deals in firm currency
            firm_currency_deals = deals[deals['Deal Currency'] == firm_currency]
            
            if len(firm_currency_deals) > 0:
                has_firm_deals += 1
            else:
                no_firm_deals += 1
                print(f"  ANOMALY: Fund {fund_id} (firm: {firm_currency}) has forward_fx_firm but no deals in firm currency")
        
        total_tested = has_firm_deals + no_firm_deals
        success_rate = has_firm_deals / total_tested * 100 if total_tested > 0 else 0
        
        print(f"  Tested {total_tested} funds")
        print(f"  Has deals in firm currency: {has_firm_deals}")
        print(f"  No deals in firm currency: {no_firm_deals}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        # Most funds with non-missing firm FX should have deals in firm currency
        self.assertGreater(
            success_rate, 80,
            f"Most funds with non-missing forward_fx_firm should have deals in firm currency, got {success_rate:.1f}%"
        )
    
    def test_edge_cases_missing_firm_fx(self):
        """Test edge cases for missing firm FX logic."""
        
        # Find funds where firm_currency == fund_currency (should rarely have missing firm FX)
        same_currency_funds = self.dta_fund[
            self.dta_fund['firm_currency'] == self.dta_fund['fund_currency']
        ]
        
        print(f"\nFound {len(same_currency_funds)} funds with same firm and fund currency")
        
        if len(same_currency_funds) > 0:
            # Check how many of these have missing firm FX measures
            same_currency_fund_ids = same_currency_funds['fund_id'].astype(str).tolist()
            
            corresponding_fx_data = self.fund_fx[
                self.fund_fx['fund_id'].astype(str).isin(same_currency_fund_ids)
            ]
            
            missing_count = corresponding_fx_data['forward_fx_firm'].isna().sum()
            total_count = len(corresponding_fx_data)
            
            missing_rate = missing_count / total_count * 100 if total_count > 0 else 0
            
            print(f"  Missing firm FX rate for same-currency funds: {missing_count}/{total_count} ({missing_rate:.1f}%)")
            
            # For same-currency funds, missing rate should be lower since they can have same-currency deals
            if total_count >= 10:  # Only test if we have enough data
                self.assertLess(
                    missing_rate, 50,
                    "Funds with same firm and fund currency should have lower missing rates"
                )
    
    def test_comprehensive_missing_logic_validation(self):
        """Comprehensive validation of the missing firm FX logic across all funds."""
        
        print(f"\nComprehensive validation summary:")
        
        # Simple validation: count missing vs non-missing and their characteristics
        total_funds_fx = len(self.fund_fx)
        missing_firm_fx = self.fund_fx['forward_fx_firm'].isna().sum()
        non_missing_firm_fx = total_funds_fx - missing_firm_fx
        
        print(f"  Total funds in fund_fx data: {total_funds_fx}")
        print(f"  Funds with missing forward_fx_firm: {missing_firm_fx} ({missing_firm_fx/total_funds_fx*100:.1f}%)")
        print(f"  Funds with non-missing forward_fx_firm: {non_missing_firm_fx} ({non_missing_firm_fx/total_funds_fx*100:.1f}%)")
        
        # Basic sanity checks
        self.assertGreater(total_funds_fx, 1000, "Should have substantial number of funds")
        self.assertGreater(non_missing_firm_fx, missing_firm_fx, "Most funds should have firm FX measures")
        self.assertLess(missing_firm_fx / total_funds_fx, 0.2, "Missing rate should be reasonable (<20%)")
        
        print("  ✓ Comprehensive validation passed")


if __name__ == '__main__':
    unittest.main(verbosity=2)
