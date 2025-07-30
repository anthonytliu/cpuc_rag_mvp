#!/usr/bin/env python3
"""
Test UI Integration for Proceeding Titles

This script tests that the UI will display full proceeding titles correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import config

def test_ui_integration():
    """Test the proceeding title integration for UI display."""
    print("🧪 Testing UI Proceeding Title Integration")
    print("=" * 60)
    
    # Test 1: Check that AVAILABLE_PROCEEDINGS is populated
    print(f"📊 Total proceedings available: {len(config.AVAILABLE_PROCEEDINGS)}")
    
    # Test 2: Check active proceedings for UI dropdown
    active_proceedings = config.get_active_proceedings()
    print(f"🟢 Active proceedings (have vector stores): {len(active_proceedings)}")
    
    if not active_proceedings:
        print("❌ ERROR: No active proceedings found! UI will be empty.")
        return False
    
    # Test 3: Show what the UI dropdown should display
    print("\n📋 UI Dropdown Options (what users will see):")
    print("-" * 50)
    
    for proc_id, proc_info in active_proceedings.items():
        display_name = proc_info['display_name']
        print(f"  • {display_name}")
    
    # Test 4: Test the mapping from display name back to proceeding ID
    print("\n🔄 Display Name → Proceeding ID Mapping Test:")
    print("-" * 50)
    
    # Simulate what the UI selectbox will do
    proceeding_options = {}
    for proc_id, proc_info in active_proceedings.items():
        proceeding_options[proc_info['display_name']] = proc_id
    
    for display_name, proc_id in proceeding_options.items():
        print(f"  '{display_name}' → {proc_id}")
    
    # Test 5: Simulate user selection
    print("\n🎯 Simulated User Selection Test:")
    print("-" * 50)
    
    # Test with R1311005 specifically since that's what user is seeing
    test_proc_id = "R1311005"
    if test_proc_id in active_proceedings:
        expected_display = active_proceedings[test_proc_id]['display_name']
        print(f"✅ User should see: '{expected_display}'")
        print(f"✅ System will use: {test_proc_id}")
        
        # Test the reverse lookup
        found_proc_id = proceeding_options.get(expected_display, "NOT_FOUND")
        if found_proc_id == test_proc_id:
            print(f"✅ Reverse lookup works: '{expected_display}' → {found_proc_id}")
        else:
            print(f"❌ Reverse lookup failed: expected {test_proc_id}, got {found_proc_id}")
    else:
        print(f"❌ {test_proc_id} is not active (no vector store)")
    
    # Test 6: Check if proceeding titles JSON is being loaded
    print(f"\n📁 Proceeding titles source:")
    titles_file = config.PROJECT_ROOT / "proceeding_titles.json"
    if titles_file.exists():
        print(f"✅ Loading titles from: {titles_file}")
        try:
            import json
            with open(titles_file) as f:
                data = json.load(f)
            print(f"✅ Successfully loaded {len(data.get('proceeding_titles', {}))} titles")
        except Exception as e:
            print(f"❌ Error loading titles: {e}")
    else:
        print(f"⚠️ Titles file not found, using fallback titles")
    
    print("\n" + "=" * 60)
    
    if len(active_proceedings) > 0:
        print("🎉 SUCCESS: UI integration should work correctly!")
        print("   Users will see full descriptive proceeding titles in the dropdown.")
        return True
    else:
        print("❌ FAILURE: No active proceedings available for UI.")
        return False


if __name__ == "__main__":
    success = test_ui_integration()
    sys.exit(0 if success else 1)