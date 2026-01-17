# ejemplo_modular.py
import pandas as pd

# Datos de ofertas
ofertas = pd.DataFrame({
    'proveedor': ['Empresa A', 'Empresa B', 'Empresa C', 'Empresa D'],
    'monto_oferta': [50_000_000, 45_000_000, 52_000_000, 48_000_000],
    'experiencia': [8, 10, 6, 12],  # años
    'metodologia': [85, 90, 75, 88],  # puntaje comisión (0-100)
    'equipo': [4, 5, 3, 6],  # profesionales
    'certificaciones': [2, 4, 1, 3]  # cantidad
})

# Crear evaluador
evaluador = EvaluadorModular()

# === CRITERIOS TÉCNICOS ===

# Experiencia: lineal, más es mejor
evaluador.agregar_criterio_tecnico(
    'experiencia',
    CriterioLineal('experiencia', peso=0.15, mayor_mejor=True)
)

# Metodología: puntaje directo de comisión
evaluador.agregar_criterio_tecnico(
    'metodologia',
    CriterioPuntajeDirecto('metodologia', peso=0.25, escala_entrada=100)
)

# Equipo: por umbrales
evaluador.agregar_criterio_tecnico(
    'equipo',
    CriterioUmbral('equipo', peso=0.10, umbrales=[
        (0, 3, 60),
        (3, 5, 80),
        (5, float('inf'), 100)
    ])
)

# Certificaciones: escalonado
evaluador.agregar_criterio_tecnico(
    'certificaciones',
    CriterioEscalonado('certificaciones', peso=0.10, escalones=[
        (0, 50),
        (2, 75),
        (3, 90),
        (4, 100)
    ])
)

# === CRITERIOS ECONÓMICOS ===

# Oferta económica: razón al mínimo
evaluador.agregar_criterio_economico(
    'monto_oferta',
    CriterioRazonMinimo('oferta_economica', peso=0.40)
)

# Evaluar
resultado = evaluador.evaluar(ofertas)

print("\n=== RESULTADO EVALUACIÓN ===")
print(resultado[[
    'proveedor', 'ranking', 'puntaje_final',
    'puntaje_tecnico_total', 'puntaje_economico_total'
]].to_string(index=False))

print("\n=== DESGLOSE DETALLADO ===")
cols_detalle = [c for c in resultado.columns if c.startswith('puntaje_')]
print(resultado[['proveedor'] + cols_detalle].to_string(index=False))

print("\n=== ESTADÍSTICOS CALCULADOS ===")
stats = evaluador.obtener_estadisticos()
for criterio, valores in stats.items():
    print(f"\n{criterio}:")
    for stat, valor in valores.items():
        print(f"  {stat}: {valor:.2f}")
