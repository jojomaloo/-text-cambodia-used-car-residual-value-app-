# Version Notes

## v2 Reference-Date Fix

This version corrects a major interpretation issue in the first MVP: the original output could be mistaken for a current 2026 market value.

Changes:

1. Added a **Valuation Reference** selector.
2. Added explicit reference labels:
   - 2024 Q3 observed market reference
   - 2025 projection based on 2024 Q3 data
   - 2026 projection based on 2024 Q3 data
   - 2027 projection based on 2024 Q3 data
3. Kept the model input vehicle age anchored to 2024 to avoid inconsistent model features.
4. Added a projection layer:
   - 2024 base value from the trained model
   - annual depreciation factor estimated from observed used-car price patterns
   - projected value = 2024 base value × depreciation factor
5. Added clear warnings that the app is not a live 2026 price feed.
6. Updated methodology, profile table, and KPI cards to show:
   - 2024 Q3 base value
   - selected projected value
   - annual depreciation assumption
   - depreciation source
   - vehicle age at valuation year
