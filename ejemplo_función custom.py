# Función custom para evaluar "cercanía al presupuesto referencial"
def evaluar_cercania_presupuesto(valores: pd.Series, stats: Dict) -> pd.Series:
    """
    Premia ofertas cercanas al presupuesto referencial
    Penaliza las muy por debajo (sospechosas) o muy por encima
    """
    presupuesto_ref = 50_000_000  # podría venir de stats o config
    
    diferencia_porcentual = abs((valores - presupuesto_ref) / presupuesto_ref) * 100
    
    # Puntaje decrece con la diferencia
    puntajes = 100 - (diferencia_porcentual * 2)
    puntajes = puntajes.clip(lower=0)  # Mínimo 0
    
    return puntajes

# Agregar criterio custom
evaluador.agregar_criterio_economico(
    'monto_oferta',
    CriterioCustom(
        'cercania_presupuesto',
        peso=0.15,
        funcion_evaluacion=evaluar_cercania_presupuesto
    )
)
