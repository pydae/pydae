from pydae.core import Builder, Model

model = Model("smib_k13p2_4ord")
model.N_store = 10000           # try tiny first
model.decimation = 10
model.ini(
    {"v_f_1": 1.0, "p_m_1": 0.0, "v_ref_2": 1.0},
    "smib_k13p2_4ord_xy_0.json",
)

model.report_u()
model.report_y()

model.run(1.0, {})
model.run(1.2, {"v_ref_2": 0.8})
model.run(2.0, {"v_ref_2":1.0})

model.post()

print(model.Time)
print(model.get_values('V_1'))
