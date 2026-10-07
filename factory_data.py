# Test factory data for the Digital Twin dashboard
# Source: case file "Цифровой двойник завода"
# These are factory KPI/demo values, separate from the AI4I sensor dataset.

production_lines = [
    {
        "date": "01.10.2026",
        "line": "Сварка-1",
        "section": "Сварка",
        "plan": 120,
        "fact": 118,
        "work_hours": 7.8,
        "load_percent": 98,
    },
    {
        "date": "01.10.2026",
        "line": "Окраска-1",
        "section": "Окраска",
        "plan": 120,
        "fact": 115,
        "work_hours": 7.5,
        "load_percent": 94,
    },
    {
        "date": "01.10.2026",
        "line": "Сборка-1",
        "section": "Сборка",
        "plan": 120,
        "fact": 121,
        "work_hours": 8.0,
        "load_percent": 100,
    },
    {
        "date": "02.10.2026",
        "line": "Сварка-1",
        "section": "Сварка",
        "plan": 120,
        "fact": 111,
        "work_hours": 7.2,
        "load_percent": 91,
    },
    {
        "date": "02.10.2026",
        "line": "Окраска-1",
        "section": "Окраска",
        "plan": 120,
        "fact": 116,
        "work_hours": 7.7,
        "load_percent": 96,
    },
    {
        "date": "02.10.2026",
        "line": "Сборка-1",
        "section": "Сборка",
        "plan": 120,
        "fact": 119,
        "work_hours": 7.9,
        "load_percent": 99,
    },
]

downtime_events = [
    {
        "date": "01.10.2026",
        "section": "Сварка",
        "equipment": "ABB-01",
        "reason": "Ошибка датчика",
        "duration_min": 25,
    },
    {
        "date": "01.10.2026",
        "section": "Окраска",
        "equipment": "Камера-02",
        "reason": "Замена фильтра",
        "duration_min": 40,
    },
    {
        "date": "02.10.2026",
        "section": "Сборка",
        "equipment": "Конвейер-03",
        "reason": "Обрыв цепи",
        "duration_min": 55,
    },
    {
        "date": "02.10.2026",
        "section": "Сварка",
        "equipment": "ABB-04",
        "reason": "Плановое ТО",
        "duration_min": 30,
    },
]

production_plan = [
    {"model": "Chevrolet Onix", "monthly_plan": 2500},
    {"model": "Chevrolet Cobalt", "monthly_plan": 1800},
    {"model": "JAC J7", "monthly_plan": 500},
]

quality_metrics = [
    {
        "date": "01.10.2026",
        "section": "Сварка",
        "produced": 118,
        "defects": 2,
        "defect_percent": 1.7,
    },
    {
        "date": "01.10.2026",
        "section": "Окраска",
        "produced": 115,
        "defects": 4,
        "defect_percent": 3.5,
    },
    {
        "date": "01.10.2026",
        "section": "Сборка",
        "produced": 121,
        "defects": 1,
        "defect_percent": 0.8,
    },
    {
        "date": "02.10.2026",
        "section": "Сварка",
        "produced": 111,
        "defects": 3,
        "defect_percent": 2.7,
    },
    {
        "date": "02.10.2026",
        "section": "Окраска",
        "produced": 116,
        "defects": 6,
        "defect_percent": 5.2,
    },
    {
        "date": "02.10.2026",
        "section": "Сборка",
        "produced": 119,
        "defects": 2,
        "defect_percent": 1.7,
    },
]

factory_flow = [
    "Склад комплектующих",
    "Сварка",
    "Окраска",
    "Сборка",
    "Контроль качества",
    "Склад готовой продукции",
]

targets = {
    "shift_hours": 8,
    "shifts_per_day": 2,
    "oee_min_percent": 85,
    "defect_max_percent": 2,
    "critical_downtime_max_min_per_day": 60,
    "monthly_output_min": 5500,
}
