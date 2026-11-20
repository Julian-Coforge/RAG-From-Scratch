"""Evaluation questions with the source section and key facts a correct answer must contain."""

EVAL_SET = [
    {
        "question": "How many days a week can I work from home?",
        "expected_source": "remote-work-policy.md",
        "expected_section": "Hybrid schedules and availability",
        "key_facts": ["two"],
    },
    {
        "question": "What is the daily meal allowance?",
        "expected_source": "expense-reimbursement-policy-v2.0.md",
        "expected_section": "Meals and incidentals",
        "key_facts": ["$70"],
    },
    {
        "question": "How long do I have to submit an expense claim?",
        "expected_source": "expense-reimbursement-policy-v2.0.md",
        "expected_section": "Receipts and submission",
        "key_facts": ["30"],
    },
    {
        "question": "When do I need an itemised receipt?",
        "expected_source": "expense-reimbursement-policy-v2.0.md",
        "expected_section": "Receipts and submission",
        "key_facts": ["$25"],
    },
    {
        "question": "How many PTO days do I get per year?",
        "expected_source": "paid-time-off-policy.md",
        "expected_section": "Entitlement and accrual",
        "key_facts": ["24"],
    },
    {
        "question": "How much unused PTO can I carry over, and until when?",
        "expected_source": "paid-time-off-policy.md",
        "expected_section": "Carryover and advance use",
        "key_facts": ["five", "31 March"],
    },
    {
        "question": "How much notice do I need to give for a week of PTO?",
        "expected_source": "paid-time-off-policy.md",
        "expected_section": "Requests and approvals",
        "key_facts": ["15"],
    },
    {
        "question": "Can I get reimbursed for extra peripherals?",
        "expected_source": "remote-work-policy.md",
        "expected_section": "Equipment and information security",
        "key_facts": ["IT service desk"],
    },
    {
        "question": "What should I do if my laptop is stolen?",
        "expected_source": "remote-work-policy.md",
        "expected_section": "Equipment and information security",
        "key_facts": ["IT service desk"],
    },
    {
        # Not covered by any document: the assistant must say so instead of guessing
        "question": "Does the company pay for a gym membership?",
        "expected_source": None,
        "expected_section": None,
        "key_facts": ["couldn't find"],
    },
]
