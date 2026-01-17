# criterios.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Callable


class CriterioBase(ABC):
    """Clase base para todos los criterios de evaluación"""
    
    def __init__(self, nombre: str, peso: float, **kwargs):
        self.nombre = nombre
        self.peso = peso
        self.config = kwargs
        self._estadisticos = {}
    
    def calcular_estadisticos(self, valores: pd.Series) -> Dict[str, Any]:
        """Calcula automáticamente estadísticos necesarios"""
        return {
            'min': valores.min(),
            'max': valores.max(),
            'mean': valores.mean(),
            'median': valores.median(),
            'std': valores.std(),
            'q25': valores.quantile(0.25),
            'q75': valores.quantile(0.75)
        }
    
    @abstractmethod
    def evaluar(self, valores: pd.Series) -> pd.Series:
        """Método abstracto para evaluar el criterio"""
        pass
    
    def normalizar(self, puntajes: pd.Series, escala: float = 100.0) -> pd.Series:
        """Normaliza puntajes a una escala (default 0-100)"""
        if puntajes.max() == puntajes.min():
            return pd.Series(escala, index=puntajes.index)
        return ((puntajes - puntajes.min()) / (puntajes.max() - puntajes.min())) * escala


class CriterioLineal(CriterioBase):
    """Normalización lineal simple"""
    
    def evaluar(self, valores: pd.Series) -> pd.Series:
        self._estadisticos = self.calcular_estadisticos(valores)
        
        # Determinar si mayor es mejor o menor es mejor
        mayor_mejor = self.config.get('mayor_mejor', True)
        
        if mayor_mejor:
            return self.normalizar(valores) * self.peso
        else:
            # Invertir: menor valor = mayor puntaje
            return self.normalizar(-valores) * self.peso


class CriterioUmbral(CriterioBase):
    """Evaluación por umbrales (rangos de puntaje)"""
    
    def evaluar(self, valores: pd.Series) -> pd.Series:
        self._estadisticos = self.calcular_estadisticos(valores)
        
        umbrales = self.config.get('umbrales', [])
        # umbrales: [(limite_inferior, limite_superior, puntaje), ...]
        
        puntajes = pd.Series(0.0, index=valores.index)
        
        for limite_inf, limite_sup, puntaje in umbrales:
            mascara = (valores >= limite_inf) & (valores < limite_sup)
            puntajes[mascara] = puntaje
        
        return puntajes * self.peso


class CriterioEscalonado(CriterioBase):
    """Puntaje escalonado según rangos"""
    
    def evaluar(self, valores: pd.Series) -> pd.Series:
        self._estadisticos = self.calcular_estadisticos(valores)
        
        escalones = self.config.get('escalones', [])
        # escalones: [(valor_minimo, puntaje), ...] ordenados ascendente
        
        puntajes = pd.Series(0.0, index=valores.index)
        
        for i, (valor_min, puntaje) in enumerate(escalones):
            mascara = valores >= valor_min
            puntajes[mascara] = puntaje
        
        return puntajes * self.peso


class CriterioPuntajeDirecto(CriterioBase):
    """Puntaje ya viene evaluado (ej: comisión evaluadora)"""
    
    def evaluar(self, valores: pd.Series) -> pd.Series:
        self._estadisticos = self.calcular_estadisticos(valores)
        
        escala_entrada = self.config.get('escala_entrada', 100)
        escala_salida = self.config.get('escala_salida', 100)
        
        # Ajustar escala si es necesario
        if escala_entrada != escala_salida:
            valores = valores * (escala_salida / escala_entrada)
        
        return valores * self.peso


class CriterioMediaGeometrica(CriterioBase):
    """Evaluación usando media geométrica (común en ofertas económicas)"""
    
    def evaluar(self, valores: pd.Series) -> pd.Series:
        self._estadisticos = self.calcular_estadisticos(valores)
        
        # Calcular media geométrica
        valores_positivos = valores[valores > 0]
        media_geo = np.exp(np.log(valores_positivos).mean())
        self._estadisticos['media_geometrica'] = media_geo
        
        # Aplicar fórmula
        puntajes = np.where(
            valores <= media_geo,
            100,
            100 - ((valores - media_geo) / media_geo) * 100
        )
        
        puntajes = np.maximum(puntajes, 0)  # No permitir negativos
        
        return pd.Series(puntajes, index=valores.index) * self.peso


class CriterioRazonMinimo(CriterioBase):
    """Puntaje = (valor_minimo / valor) * 100"""
    
    def evaluar(self, valores: pd.Series) -> pd.Series:
        self._estadisticos = self.calcular_estadisticos(valores)
        
        valor_min = valores.min()
        puntajes = (valor_min / valores) * 100
        
        return puntajes * self.peso


class CriterioInverso(CriterioBase):
    """Puntaje inversamente proporcional"""
    
    def evaluar(self, valores: pd.Series) -> pd.Series:
        self._estadisticos = self.calcular_estadisticos(valores)
        
        inversas = 1 / valores
        puntajes = (inversas / inversas.sum()) * 100
        
        return puntajes * self.peso


class CriterioCustom(CriterioBase):
    """Permite función personalizada de evaluación"""
    
    def __init__(self, nombre: str, peso: float, 
                 funcion_evaluacion: Callable[[pd.Series, Dict], pd.Series],
                 **kwargs):
        super().__init__(nombre, peso, **kwargs)
        self.funcion_evaluacion = funcion_evaluacion
    
    def evaluar(self, valores: pd.Series) -> pd.Series:
        self._estadisticos = self.calcular_estadisticos(valores)
        
        # La función custom recibe los valores y los estadísticos
        puntajes = self.funcion_evaluacion(valores, self._estadisticos)
        
        return puntajes * self.peso


# evaluator.py
class EvaluadorModular:
    """Motor de evaluación modular"""
    
    def __init__(self):
        self.criterios_tecnicos: Dict[str, CriterioBase] = {}
        self.criterios_economicos: Dict[str, CriterioBase] = {}
        self.ponderacion_tecnica: float = 0.0
        self.ponderacion_economica: float = 0.0
    
    def agregar_criterio_tecnico(self, columna: str, criterio: CriterioBase):
        """Agrega un criterio técnico"""
        self.criterios_tecnicos[columna] = criterio
        self._recalcular_ponderaciones()
    
    def agregar_criterio_economico(self, columna: str, criterio: CriterioBase):
        """Agrega un criterio económico"""
        self.criterios_economicos[columna] = criterio
        self._recalcular_ponderaciones()
    
    def _recalcular_ponderaciones(self):
        """Recalcula ponderaciones automáticamente"""
        peso_tec = sum(c.peso for c in self.criterios_tecnicos.values())
        peso_eco = sum(c.peso for c in self.criterios_economicos.values())
        
        total = peso_tec + peso_eco
        
        if total > 0:
            self.ponderacion_tecnica = peso_tec / total
            self.ponderacion_economica = peso_eco / total
    
    def evaluar(self, ofertas_df: pd.DataFrame) -> pd.DataFrame:
        """Evalúa todas las ofertas"""
        resultado = ofertas_df.copy()
        
        # Evaluar criterios técnicos
        puntajes_tecnicos = []
        for columna, criterio in self.criterios_tecnicos.items():
            puntaje = criterio.evaluar(ofertas_df[columna])
            resultado[f'puntaje_{criterio.nombre}'] = puntaje
            puntajes_tecnicos.append(puntaje)
        
        if puntajes_tecnicos:
            resultado['puntaje_tecnico_total'] = sum(puntajes_tecnicos)
        else:
            resultado['puntaje_tecnico_total'] = 0
        
        # Evaluar criterios económicos
        puntajes_economicos = []
        for columna, criterio in self.criterios_economicos.items():
            puntaje = criterio.evaluar(ofertas_df[columna])
            resultado[f'puntaje_{criterio.nombre}'] = puntaje
            puntajes_economicos.append(puntaje)
        
        if puntajes_economicos:
            resultado['puntaje_economico_total'] = sum(puntajes_economicos)
        else:
            resultado['puntaje_economico_total'] = 0
        
        # Puntaje final
        resultado['puntaje_final'] = (
            resultado['puntaje_tecnico_total'] * self.ponderacion_tecnica +
            resultado['puntaje_economico_total'] * self.ponderacion_economica
        )
        
        # Ranking
        resultado['ranking'] = resultado['puntaje_final'].rank(
            ascending=False, method='min'
        )
        
        return resultado.sort_values('ranking')
    
    def obtener_estadisticos(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene estadísticos calculados de todos los criterios"""
        estadisticos = {}
        
        for columna, criterio in {**self.criterios_tecnicos, 
                                   **self.criterios_economicos}.items():
            if criterio._estadisticos:
                estadisticos[criterio.nombre] = criterio._estadisticos
        
        return estadisticos
