import random
from models import (
    AccountProfile, Transaction, LoginEvent,
    AccountEvent, FraudObservation
)
_case_queues = {}
def get_case(task: str, case_id: str = None) -> dict:

    if task == "task_easy":
        cases = [
            _case_geo_impossible,
            _case_velocity_fraud,
            _case_legitimate_traveler,
        ]
    elif task == "task_medium":
        cases = [
            _case_device_takeover,
            _case_credential_stuffing,
            _case_legitimate_new_device,
        ]
    elif task == "task_hard":
        cases = [
            _case_money_mule_network,
            _case_bust_out_fraud,
            _case_legitimate_business,
        ]
    else:
        raise ValueError(f"Unknown task: {task}")

    if task not in _case_queues or not _case_queues[task]:
        _case_queues[task] = cases[:]
        random.shuffle(_case_queues[task])

    case_fn = _case_queues[task].pop()
    return case_fn()

# ─────────────────────────────────────────
# TASK 1: CREDIT CARD FRAUD CASES
# ─────────────────────────────────────────

def _case_geo_impossible():
      obs = FraudObservation(
          case_id="CC-001",
          task="task_easy",
          step=0,
          account=AccountProfile(
              account_id="ACC-1001",
              name="Priya Sharma",
              email="priya.sharma@gmail.com",
              location="Mumbai, India",
              account_age_days=1240,
              avg_monthly_spend=18000.0,
              usual_merchants=["Big Bazaar", "Reliance Fresh", "HPCL Fuel", "Zomato"],
              credit_limit=50000.0,
              ssn_last4=None
          ),
          transactions=[
              Transaction(txn_id="T001", amount=450.0, merchant="Big Bazaar",
                         category="grocery", location="Mumbai, India",
                         timestamp="2024-03-15 09:14:00", card_present=True),
              Transaction(txn_id="T002", amount=1200.0, merchant="HPCL Fuel",
                         category="fuel", location="Mumbai, India",
                         timestamp="2024-03-15 11:30:00", card_present=True),
              # suspicious one
              Transaction(txn_id="T003", amount=48000.0, merchant="Electronics Hub",
                         category="electronics", location="Lagos, Nigeria",
                         timestamp="2024-03-15 15:14:00", card_present=False),
          ],
          login_events=[],
          account_events=[],
          linked_accounts=[],
          additional_signals={
              "distance_km_between_last_two_txns": 8200,
              "time_between_last_two_txns_mins": 224,
              "speed_required_kmh": 2196,
              "is_card_present": False,
              "time_of_day": "late_night_local",
              "merchant_category_new": True
          },
          reward=0.0,
          done=False,
          feedback=""
      )
      truth = {
          "is_fraud": True,
          "fraud_type": "card_fraud",
          "attack_vector": "geo_impossible",
          "key_signals": ["impossible_travel", "max_limit_attempt",
                          "new_merchant_category", "card_not_present",
                          "foreign_country"],
          "action": "block_card",
          "acceptable_actions": ["block_card", "hold_for_review"],
          "regulatory_action": "none"
      }
      return {"observation": obs, "truth": truth}


def _case_velocity_fraud():
    obs = FraudObservation(
        case_id="CC-002",
        task="task_easy",
        step=0,
        account=AccountProfile(
            account_id="ACC-1002",
            name="Rahul Mehta",
            email="rahul.m@yahoo.com",
            location="Bangalore, India",
            account_age_days=890,
            avg_monthly_spend=12000.0,
            usual_merchants=["Swiggy", "Amazon", "Flipkart"],
            credit_limit=30000.0,
            ssn_last4=None
        ),
        transactions=[
            Transaction(txn_id="T010", amount=890.0, merchant="Amazon",
                        category="ecommerce", location="Online",
                        timestamp="2024-03-20 11:02:00", card_present=False),
            Transaction(txn_id="T011", amount=1560.0, merchant="Best Electronics",
                        category="electronics", location="Online",
                        timestamp="2024-03-20 11:04:00", card_present=False),
            Transaction(txn_id="T012", amount=2100.0, merchant="Luxury Watches",
                        category="jewelry", location="Online",
                        timestamp="2024-03-20 11:06:00", card_present=False),
            Transaction(txn_id="T013", amount=4450.0, merchant="Apple Store",
                        category="electronics", location="Online",
                        timestamp="2024-03-20 11:07:00", card_present=False),
            Transaction(txn_id="T014", amount=8990.0, merchant="Gold Jewellers",
                        category="jewelry", location="Online",
                        timestamp="2024-03-20 11:09:00", card_present=False),
        ],
        login_events=[],
        account_events=[],
        linked_accounts=[],
        additional_signals={
            "txns_in_last_10_mins": 5,
            "total_amount_10_mins": 17990.0,
            "avg_txn_per_day_historically": 1.2,
            "escalating_amounts": True,
            "all_card_not_present": True
        },
        reward=0.0,
        done=False,
        feedback=""
    )
    truth = {
        "is_fraud": True,
        "fraud_type": "card_fraud",
        "attack_vector": "velocity",
        "key_signals": ["velocity_spike", "escalating_amounts",
                        "card_not_present", "multiple_new_categories",
                        "high_value_merchants"],
        "action": "block_card",
        "acceptable_actions": ["block_card", "hold_for_review"],
        "regulatory_action": "none"
    }
    return {"observation": obs, "truth": truth}


def _case_legitimate_traveler():
    obs = FraudObservation(
        case_id="CC-003",
        task="task_easy",
        step=0,
        account=AccountProfile(
            account_id="ACC-1003",
            name="Ananya Iyer",
            email="ananya.iyer@gmail.com",
            location="Chennai, India",
            account_age_days=2100,
            avg_monthly_spend=45000.0,
            usual_merchants=["Airlines", "Hotels", "Restaurants"],
            credit_limit=200000.0,
            ssn_last4=None
        ),
        transactions=[
            Transaction(txn_id="T020", amount=85000.0, merchant="British Airways",
                        category="travel", location="Online",
                        timestamp="2024-03-10 14:00:00", card_present=False),
            Transaction(txn_id="T021", amount=12000.0, merchant="Travel Insurance Co",
                        category="insurance", location="Online",
                        timestamp="2024-03-10 14:05:00", card_present=False),
            Transaction(txn_id="T022", amount=15000.0, merchant="Harrods",
                        category="shopping", location="London, UK",
                        timestamp="2024-03-15 16:30:00", card_present=True),
        ],
        login_events=[],
        account_events=[],
        linked_accounts=[],
        additional_signals={
            "flight_booking_found": True,
            "destination_matches_txn_location": True,
            "user_is_frequent_traveler": True,
            "amount_within_normal_range": True
        },
        reward=0.0,
        done=False,
        feedback=""
    )
    truth = {
        "is_fraud": False,
        "fraud_type": "legitimate",
        "attack_vector": "none",
        "key_signals": ["flight_booking_found",
                        "destination_matches_txn_location",
                        "frequent_traveler_profile"],
        "action": "allow",
        "acceptable_actions": ["allow"],
        "regulatory_action": "none"
    }
    return {"observation": obs, "truth": truth}


def _case_card_not_present():
    obs = FraudObservation(
        case_id="CC-004",
        task="task_easy",
        step=0,
        account=AccountProfile(
            account_id="ACC-1004",
            name="Vikram Nair",
            email="vikram.nair@gmail.com",
            location="Delhi, India",
            account_age_days=650,
            avg_monthly_spend=8000.0,
            usual_merchants=["Grocery", "Fuel", "Local shops"],
            credit_limit=25000.0,
            ssn_last4=None
        ),
        transactions=[
            Transaction(txn_id="T030", amount=6500.0, merchant="Online Tech Store",
                        category="electronics", location="Online",
                        timestamp="2024-03-18 02:30:00", card_present=False),
        ],
        login_events=[],
        account_events=[],
        linked_accounts=[],
        additional_signals={
            "billing_address_match": True,
            "shipping_address": "Freight Forwarder, Mumbai",
            "shipping_is_freight_forwarder": True,
            "device_ip": "185.220.101.42",
            "ip_is_vpn": True,
            "ip_country": "Russia",
            "email_account_age_days": 3,
            "time_of_day": "2:30 AM local"
        },
        reward=0.0,
        done=False,
        feedback=""
    )
    truth = {
        "is_fraud": True,
        "fraud_type": "card_fraud",
        "attack_vector": "card_not_present",
        "key_signals": ["freight_forwarder", "vpn_ip", "new_email",
                        "late_night", "foreign_ip"],
        "action": "hold_for_review",
        "acceptable_actions": ["hold_for_review", "block_card"],
        "regulatory_action": "none"
    }
    return {"observation": obs, "truth": truth}


# ─────────────────────────────────────────
# TASK 2: ACCOUNT TAKEOVER CASES
# ─────────────────────────────────────────

def _case_device_takeover():
      obs = FraudObservation(
          case_id="ATO-001",
          task="task_medium",
          step=0,
          account=AccountProfile(
              account_id="ACC-2001",
              name="Suresh Kumar",
              email="suresh.k@gmail.com",
              location="Hyderabad, India",
              account_age_days=1800,
              avg_monthly_spend=22000.0,
              usual_merchants=["Amazon", "Swiggy", "Netflix"],
              credit_limit=100000.0,
              ssn_last4=None
          ),
          transactions=[
              Transaction(txn_id="T040", amount=45000.0, merchant="Wire Transfer",
                         category="transfer", location="Online",
                         timestamp="2024-03-22 02:20:00", card_present=False),
              Transaction(txn_id="T041", amount=32000.0, merchant="Wire Transfer",
                         category="transfer", location="Online",
                         timestamp="2024-03-22 02:21:00", card_present=False),
          ],
          login_events=[
              LoginEvent(timestamp="2024-03-22 02:13:00", device="Android Samsung (NEW)",
                        ip_address="185.x.x.x", location="Netherlands (VPN)",
                        success=True, note="Device never seen before"),
          ],
          account_events=[
              AccountEvent(timestamp="2024-03-22 02:15:00", event_type="password_change",
                          old_value="[hidden]", new_value="[hidden]"),
              AccountEvent(timestamp="2024-03-22 02:16:00", event_type="email_change",
                          old_value="suresh.k@gmail.com", new_value="sureshnew123@gmail.com"),
              AccountEvent(timestamp="2024-03-22 02:17:00", event_type="phone_change",
                          old_value="+91-98xxx-xxxxx", new_value="+1-555-000-1234"),
          ],
          linked_accounts=[],
          additional_signals={
              "usual_device": "MacBook Pro (seen 847 times)",
              "usual_login_time": "9AM-11PM",
              "new_payee_added": True,
              "transfers_to_new_payee": True,
              "all_changes_within_8_mins": True
          },
          reward=0.0,
          done=False,
          feedback=""
      )
      truth = {
          "is_fraud": True,
          "fraud_type": "account_takeover",
          "attack_vector": "credential_compromise",
          "key_signals": ["new_device", "vpn_ip", "password_change",
                          "email_change", "phone_change", "large_transfers",
                          "new_payee", "2am_activity", "all_changes_rapid"],
          "action": "freeze_account",
          "acceptable_actions": ["freeze_account", "escalate"],
          "regulatory_action": "none"
      }
      return {"observation": obs, "truth": truth}


def _case_credential_stuffing():
    obs = FraudObservation(
        case_id="ATO-002",
        task="task_medium",
        step=0,
        account=AccountProfile(
            account_id="ACC-2002",
            name="Meera Pillai",
            email="meera.pillai@hotmail.com",
            location="Kochi, India",
            account_age_days=920,
            avg_monthly_spend=15000.0,
            usual_merchants=["Flipkart", "BookMyShow", "Local restaurants"],
            credit_limit=60000.0,
            ssn_last4=None
        ),
        transactions=[
            Transaction(txn_id="T050", amount=20000.0, merchant="Wire Transfer",
                       category="transfer", location="Online",
                       timestamp="2024-03-25 23:40:00", card_present=False),
        ],
        login_events=[
            LoginEvent(timestamp="2024-03-25 22:00:00", device="Various",
                      ip_address="Multiple (23 IPs)", location="12 countries",
                      success=False, note="847 failed attempts over 90 mins"),
            LoginEvent(timestamp="2024-03-25 23:34:00", device="Chrome Windows",
                      ip_address="89.x.x.x", location="Romania",
                      success=True, note="Success after spray attack"),
        ],
        account_events=[
            AccountEvent(timestamp="2024-03-25 23:36:00", event_type="address_change",
                        old_value="Kochi, Kerala", new_value="123 Fake St, Bucharest"),
        ],
        linked_accounts=[],
        additional_signals={
            "email_found_in_breaches": 3,
            "password_reused_from_breach": True,
            "breach_source": "LinkedIn 2021 breach",
            "failed_logins_before_success": 847,
            "ips_used": 23,
            "countries_involved": 12
        },
        reward=0.0,
        done=False,
        feedback=""
    )
    truth = {
        "is_fraud": True,
        "fraud_type": "account_takeover",
        "attack_vector": "credential_stuffing",
        "key_signals": ["login_spray", "breach_database_match",
                        "password_reuse", "foreign_successful_login",
                        "immediate_transfer", "address_change"],
        "action": "freeze_account",
        "acceptable_actions": ["freeze_account", "escalate"],
        "regulatory_action": "none"
    }
    return {"observation": obs, "truth": truth}


def _case_sim_swap():
      obs = FraudObservation(
          case_id="ATO-003",
          task="task_medium",
          step=0,
          account=AccountProfile(
              account_id="ACC-2003",
              name="Arjun Reddy",
              email="arjun.reddy@gmail.com",
              location="Pune, India",
              account_age_days=1450,
              avg_monthly_spend=35000.0,
              usual_merchants=["Amazon", "Zomato", "MakeMyTrip"],
              credit_limit=150000.0,
              ssn_last4=None
          ),
          transactions=[
              Transaction(txn_id="T060", amount=50000.0, merchant="Wire Transfer",
                         category="transfer", location="Online",
                         timestamp="2024-03-28 10:25:00", card_present=False),
              Transaction(txn_id="T061", amount=45000.0, merchant="Wire Transfer",
                         category="transfer", location="Online",
                         timestamp="2024-03-28 10:26:00", card_present=False),
              Transaction(txn_id="T062", amount=40000.0, merchant="Wire Transfer",
                         category="transfer", location="Online",
                         timestamp="2024-03-28 10:27:00", card_present=False),
          ],
          login_events=[
              LoginEvent(timestamp="2024-03-28 10:19:00", device="New Device",
                        ip_address="Unknown", location="Unknown",
                        success=True, note="OTP used for auth"),
          ],
          account_events=[],
          linked_accounts=[],
          additional_signals={
              "sim_swap_detected": True,
              "sim_swap_time": "2024-03-28 10:02:00",
              "minutes_between_sim_swap_and_login": 17,
              "carrier": "Airtel",
              "otp_sent_to_new_sim": True,
              "total_drained": 135000.0
          },
          reward=0.0,
          done=False,
          feedback=""
      )
      truth = {
          "is_fraud": True,
          "fraud_type": "account_takeover",
          "attack_vector": "sim_swap",
          "key_signals": ["sim_swap_event", "otp_post_sim_swap",
                          "rapid_drain", "new_device", "large_transfers"],
          "action": "freeze_account",
          "acceptable_actions": ["freeze_account", "escalate"],
          "regulatory_action": "none"
      }
      return {"observation": obs, "truth": truth}


def _case_legitimate_new_device():
      obs = FraudObservation(
          case_id="ATO-004",
          task="task_medium",
          step=0,
          account=AccountProfile(
              account_id="ACC-2004",
              name="Pooja Desai",
              email="pooja.desai@gmail.com",
              location="Mumbai, India",
              account_age_days=2400,
              avg_monthly_spend=20000.0,
              usual_merchants=["Amazon", "Swiggy", "Big Bazaar"],
              credit_limit=80000.0,
              ssn_last4=None
          ),
          transactions=[
              Transaction(txn_id="T070", amount=3200.0, merchant="Amazon",
                         category="ecommerce", location="Online",
                         timestamp="2024-03-30 19:30:00", card_present=False),
          ],
          login_events=[
              LoginEvent(timestamp="2024-03-30 19:15:00", device="iPhone 15 (NEW)",
                        ip_address="103.x.x.x", location="Mumbai, India",
                        success=True, note="OTP verified successfully"),
          ],
          account_events=[
              AccountEvent(timestamp="2024-03-30 19:20:00", event_type="address_change",
                          old_value="Andheri West, Mumbai", new_value="Bandra West, Mumbai"),
          ],
          linked_accounts=[],
          additional_signals={
              "login_time_normal": True,
              "location_same_city": True,
              "merchant_is_usual": True,
              "amount_within_normal_range": True,
              "user_support_call_yesterday": "Reported old phone stolen, asked for new device setup",
              "amount_consistent_with_history": True
          },
          reward=0.0,
          done=False,
          feedback=""
      )
      truth = {
          "is_fraud": False,
          "fraud_type": "legitimate",
          "attack_vector": "none",
          "key_signals": ["support_call_context", "same_city",
                          "normal_time", "usual_merchant"],
          "action": "allow",
          "acceptable_actions": ["allow"],
          "regulatory_action": "none"
      }
      return {"observation": obs, "truth": truth}


# ─────────────────────────────────────────
# TASK 3: NETWORK FRAUD CASES
# ─────────────────────────────────────────


def _case_money_mule_network():
    linked = [
        AccountProfile(account_id="ACC-3001", name="John Smith",
                       email="jsmith1@gmail.com", location="Delhi",
                       account_age_days=95, avg_monthly_spend=0,
                       usual_merchants=[], credit_limit=None, ssn_last4="4521"),
        AccountProfile(account_id="ACC-3002", name="Jon Smyth",
                       email="jsmyth2@gmail.com", location="Delhi",
                       account_age_days=92, avg_monthly_spend=0,
                       usual_merchants=[], credit_limit=None, ssn_last4="4521"),
        AccountProfile(account_id="ACC-3003", name="J. Smith",
                       email="jsmith3@yahoo.com", location="Delhi",
                       account_age_days=90, avg_monthly_spend=0,
                       usual_merchants=[], credit_limit=None, ssn_last4="4521"),
        AccountProfile(account_id="ACC-3004", name="Jane Smythe",
                       email="jsmythe4@gmail.com", location="Delhi",
                       account_age_days=88, avg_monthly_spend=0,
                       usual_merchants=[], credit_limit=None, ssn_last4="4521"),
        AccountProfile(account_id="ACC-3005", name="John Smoth",
                       email="jsmoth5@gmail.com", location="Delhi",
                       account_age_days=85, avg_monthly_spend=0,
                       usual_merchants=[], credit_limit=None, ssn_last4="4521"),
    ]
    obs = FraudObservation(
        case_id="NET-001",
        task="task_hard",
        step=0,
        account=linked[0],
        transactions=[
            Transaction(txn_id="T080", amount=195.0, merchant="Deposit",
                        category="deposit", location="ATM Delhi",
                        timestamp="2024-01-10 10:00:00", card_present=True),
            Transaction(txn_id="T081", amount=210.0, merchant="Deposit",
                        category="deposit", location="ATM Delhi",
                        timestamp="2024-02-10 10:00:00", card_present=True),
            Transaction(txn_id="T082", amount=205.0, merchant="Deposit",
                        category="deposit", location="ATM Delhi",
                        timestamp="2024-03-10 10:00:00", card_present=True),
            Transaction(txn_id="T083", amount=14300.0, merchant="Wire Transfer OUT",
                        category="transfer", location="Online",
                        timestamp="2024-03-15 14:00:00", card_present=False),
        ],
        login_events=[],
        account_events=[],
        linked_accounts=linked[1:],
        additional_signals={
            "all_accounts_same_device_fingerprint": True,
            "devices_used": ["Device-A", "Device-B"],
            "all_accounts_opened_within_days": 10,
            "same_ssn_across_accounts": True,
            "ssn_belongs_to_minor_aged_8": True,
            "all_transfers_same_day": "2024-03-15",
            "all_transfers_within_hours": 2,
            "destination_account": "EXT-9987",
            "total_extracted": 14300.0,
            "below_reporting_threshold_deposits": True,
            "structuring_detected": True
        },
        reward=0.0,
        done=False,
        feedback=""
    )
    truth = {
        "is_fraud": True,
        "fraud_type": "money_mule",
        "attack_vector": "synthetic_identity_network",
        "key_signals": ["same_ssn_variants", "ssn_belongs_to_minor",
                        "structured_deposits", "same_device",
                        "coordinated_extraction", "new_accounts", "no_spending"],
        "action": "freeze_account",
        "acceptable_actions": ["freeze_account", "file_SAR"],
        "regulatory_action": "SAR",
        "network_accounts": ["ACC-3001", "ACC-3002", "ACC-3003", "ACC-3004", "ACC-3005"],
        "hub_account": "EXT-9987"
    }
    return {"observation": obs, "truth": truth}


def _case_bust_out_fraud():
    obs = FraudObservation(
        case_id="NET-002",
        task="task_hard",
        step=0,
        account=AccountProfile(
            account_id="ACC-3010", name="Ravi Verma",
            email="ravi.v@gmail.com", location="Jaipur, India",
            account_age_days=240, avg_monthly_spend=5000.0,
            usual_merchants=["Local shops"], credit_limit=80000.0,
            ssn_last4="7823"
        ),
        transactions=[
            Transaction(txn_id="T090", amount=2000.0, merchant="Various",
                        category="misc", location="Jaipur",
                        timestamp="2024-01-01 00:00:00", card_present=True),
            Transaction(txn_id="T091", amount=75000.0, merchant="Cash Advance",
                        category="cash", location="ATM",
                        timestamp="2024-03-20 09:00:00", card_present=True),
            Transaction(txn_id="T092", amount=80000.0, merchant="Jewellery Store",
                        category="jewelry", location="Jaipur",
                        timestamp="2024-03-21 11:00:00", card_present=True),
        ],
        login_events=[],
        account_events=[
            AccountEvent(timestamp="2024-03-22 00:00:00",
                         event_type="address_change",
                         old_value="12 MG Road, Jaipur",
                         new_value="Vacant plot, outskirts"),
        ],
        linked_accounts=[],
        additional_signals={
            "credit_score_trend": "580 → 720 over 6 months",
            "credit_limit_increases": 3,
            "payment_history": "Always paid minimum, on time",
            "sudden_maxout_days": 4,
            "new_address_is_vacant_lot": True,
            "phone_disconnected": True,
            "similar_accounts_pattern_match": 11,
            "all_similar_accounts_opened_same_week": True,
            "total_owed": 155000.0
        },
        reward=0.0,
        done=False,
        feedback=""
    )
    truth = {
        "is_fraud": True,
        "fraud_type": "bust_out",
        "attack_vector": "organized_bust_out",
        "key_signals": ["slow_buildup", "sudden_maxout", "address_vacant",
                        "phone_disconnected", "coordinated_with_others",
                        "cash_advance_spike"],
        "action": "freeze_account",
        "acceptable_actions": ["freeze_account", "file_SAR"],
        "regulatory_action": "SAR",
        "network_accounts": ["ACC-3010"],
        "hub_account": None
    }
    return {"observation": obs, "truth": truth}


def _case_legitimate_business():
    linked = [
        AccountProfile(account_id="BIZ-001", name="TechStart Pvt Ltd - Payroll",
                       email="accounts@techstart.in", location="Bangalore",
                       account_age_days=730, avg_monthly_spend=500000.0,
                       usual_merchants=["Payroll", "Vendors"], credit_limit=None, ssn_last4=None),
        AccountProfile(account_id="BIZ-002", name="TechStart Pvt Ltd - Operations",
                       email="ops@techstart.in", location="Bangalore",
                       account_age_days=720, avg_monthly_spend=200000.0,
                       usual_merchants=["AWS", "Office supplies"], credit_limit=None, ssn_last4=None),
    ]
    obs = FraudObservation(
        case_id="NET-003",
        task="task_hard",
        step=0,
        account=AccountProfile(
            account_id="BIZ-000", name="TechStart Pvt Ltd - Main",
            email="finance@techstart.in", location="Bangalore",
            account_age_days=750, avg_monthly_spend=1000000.0,
            usual_merchants=["Payroll", "AWS", "Office", "Vendors"],
            credit_limit=None, ssn_last4=None
        ),
        transactions=[
            Transaction(txn_id="T100", amount=450000.0, merchant="Payroll Transfer",
                        category="payroll", location="Online",
                        timestamp="2024-03-31 10:00:00", card_present=False),
            Transaction(txn_id="T101", amount=180000.0, merchant="AWS Invoice",
                        category="cloud", location="Online",
                        timestamp="2024-03-31 10:05:00", card_present=False),
        ],
        login_events=[],
        account_events=[],
        linked_accounts=linked,
        additional_signals={
            "business_registration": "Registered 2022, MCA verified",
            "gst_filing_current": True,
            "tax_returns_filed": True,
            "all_transfers_have_invoice": True,
            "same_day_transfers_monthly": True,
            "pattern_is_payroll_cycle": True,
            "audited_account": True
        },
        reward=0.0,
        done=False,
        feedback=""
    )
    truth = {
        "is_fraud": False,
        "fraud_type": "legitimate",
        "attack_vector": "none",
        "key_signals": ["business_registration", "gst_filing",
                        "invoice_backed_transfers", "consistent_payroll_pattern"],
        "action": "allow",
        "acceptable_actions": ["allow"],
        "regulatory_action": "none",
        "network_accounts": [],
        "hub_account": None
    }
    return {"observation": obs, "truth": truth}
