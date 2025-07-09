"""
Configuration management for ragpas package.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Initialize logging
logger = logging.getLogger(__name__)

try:
    import tomllib
except ImportError:
    import logging
    logger.error("Neither tomllib nor tomli is available. Please install tomli for Python < 3.11")
    raise ImportError("TOML library not available")

try:
    import tomli_w
except ImportError:
    import logging
    logger.error("Neither tomli_w nor toml is available for writing TOML files")
    raise ImportError("TOML writing library not available")


@dataclass
class BaseConfig(ABC):
    """Base configuration class for all ragpas configurations."""
    
    # Common configuration
    log_level: str = "INFO"

    proxy: Optional[str] = None

    _initialize: bool = False
    
    @classmethod
    @abstractmethod
    def get_config_section(cls) -> str:
        """Get the configuration section name in TOML file."""
        pass

    def initialize(self) -> None:
        """Initialize the configuration."""
        self._initialize = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Create configuration from dictionary."""
        # Filter only valid attributes
        valid_attrs = {k: v for k, v in config_dict.items() if hasattr(cls, k)}
        return cls(**valid_attrs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
            if getattr(self, field.name) is not None
        }
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")

@dataclass
class GlobalConfig(BaseConfig):
    """Configuration for global settings."""

    @classmethod
    def get_config_section(cls) -> str:
        return "global"

@dataclass
class MIAConfig(BaseConfig):
    """Configuration for MIA (Membership Inference Attack) module."""
    
    # Model configuration
    default_model: str = None
    extract_model: str = None
    generate_model: str = None
    evaluate_model: str = None


    # Input/Output configuration
    dataset_input_path: str = None
    mia_output_path: str = None
    calculation_output_path: str = None


    # Target configuration
    target: str = None

    # Processing configuration
    save_step: int = 10

    @classmethod
    def get_config_section(cls) -> str:
        return "mia"


    def initialize(self) -> None:
        """Validate input and output file paths."""
        if not self.dataset_input_path:
            raise ValueError("dataset_input_path must be specified")

        if not self.mia_output_path:
            raise ValueError("mia_output_path must be specified")

        if not self.calculation_output_path:
            raise ValueError("calculation_output_path must be specified")

        # Handle directory paths
        if os.path.isdir(self.dataset_input_path):
            input_file = os.path.join(self.dataset_input_path, "input.csv")
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")
            self.dataset_input_path = input_file
        elif not os.path.exists(self.dataset_input_path):
            raise FileNotFoundError(f"Input file not found: {self.dataset_input_path}")

        if os.path.isdir(self.mia_output_path):
            self.mia_output_path = os.path.join(self.mia_output_path, "mia_output.csv")
        if os.path.exists(self.mia_output_path):
            newfilename = os.path.splitext(self.mia_output_path)[0] + "_new.csv"
            logger.error(f"Output file already exists: {self.mia_output_path}, change the output file name to {newfilename}")
            self.mia_output_path = newfilename

        output_dir = os.path.dirname(self.mia_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if os.path.isdir(self.calculation_output_path):
            self.calculation_output_path = os.path.join(self.calculation_output_path, "result.csv")
        if os.path.exists(self.calculation_output_path):
            newfilename = os.path.splitext(self.calculation_output_path)[0] + "_new.csv"
            logger.error(f"Output file already exists: {self.calculation_output_path}, change the output file name to {newfilename}")
            self.calculation_output_path = newfilename

        self._initialize = True

@dataclass
class KPAConfig(BaseConfig):
    """Configuration for KPA (Key Point Attack) module."""
    
    # Model configuration
    model: str = "doubao-1-5-lite"
    
    # Input/Output configuration
    input_filename: str = "kpa_input.csv"
    output_filename: str = "kpa_output.csv"
    input_file_path: str = ""
    output_file_path: str = ""
    
    # Target configuration
    target_entities: list = field(default_factory=lambda: ["Person", "Organization"])
    
    # Processing configuration
    save_step: int = 10
    batch_size: int = 32
    confidence_threshold: float = 0.8
    
    @classmethod
    def get_config_section(cls) -> str:
        return "kpa"


@dataclass 
class DoSAConfig(BaseConfig):
    """Configuration for DoSA (Data Ownership Security Attack) module."""

    # Model configuration
    model: str = "doubao-seed-1-6-flash-250615"
    
    # Input/Output configuration
    input_filename: str = "dosa_input.csv"
    output_filename: str = "dosa_output.csv"
    input_file_path: str = ""
    output_file_path: str = ""
    
    # Processing configuration
    save_step: int = 10
    batch_size: int = 32
    attack_strategies: list = field(default_factory=lambda: ["direct", "indirect"])
    max_iterations: int = 100
    
    @classmethod
    def get_config_section(cls) -> str:
        return "dosa"


@dataclass
class RAGConfig(BaseConfig):
    """Configuration for RAG (Retrieval-Augmented Generation) module."""
    
    # Database configuration
    database_path: str = None
    collection_name: str = "retrieval_database"
    
    # Model configuration
    llm_model: str = None
    embedding_model: str = None
    embedding_dimensions: Optional[int] = None
    
    # Retrieval configuration
    chunk_size: int = 4000
    chunk_overlap: int = 200
    top_k: int = 5
    
    # Generation configuration
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    
    @classmethod
    def get_config_section(cls) -> str:
        return "rag"

    def initialize(self) -> None:
        """Validate and initialize RAG configuration."""
        if not self.database_path:
            raise ValueError("database_path must be specified")
        
        if not self.llm_model:
            raise ValueError("llm_model must be specified")
        
        if not self.embedding_model:
            raise ValueError("embedding_model must be specified")
        
        if self.embedding_dimensions is None:
            raise ValueError("embedding_dimensions must be specified")
        
        # Ensure database path exists
        db_path = Path(self.database_path)
        if not db_path.exists():
            db_path.mkdir(parents=True, exist_ok=True)
        
        self._initialize = True


class ConfigManager:
    """Centralized configuration manager for ragpas package."""
    
    # Registry of configuration classes
    _config_classes = {
        "global": GlobalConfig,
        "mia": MIAConfig,
        "kpa": KPAConfig,
        "dosa": DoSAConfig,
        "rag": RAGConfig,
    }
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize config manager.
        
        Args:
            config_dir: Directory containing configuration files. 
                       Defaults to project root directory.
        """
        if config_dir is None:
            # Find project root (where pyproject.toml is located)
            current_dir = Path(__file__).parent
            while current_dir != current_dir.parent:
                if (current_dir / "pyproject.toml").exists():
                    self.config_dir = current_dir
                    break
                current_dir = current_dir.parent
            else:
                # Fallback to current directory
                self.config_dir = Path.cwd()
        else:
            self.config_dir = Path(config_dir)
        
        self._configs = {}  # Cache for loaded configurations
    
    def load_config_from_toml(self, config_file: str = "config.toml") -> Dict[str, BaseConfig]:
        """Load all configurations from TOML file."""
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}. Using default configurations.")
            raise FileNotFoundError(f"Config file {config_path} does not exist.")
                
        try:
            with open(config_path, "rb") as f:
                config_data = tomllib.load(f)
            configs = {}
            for config_name, config_class in self._config_classes.items():
                section_data: dict[str, Any] = config_data.get(config_name, {})
                section_data.update(config_data.get("global", {}))
                configs[config_name] = config_class.from_dict(section_data)
            
            return configs
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise ValueError(f"Failed to load configurations from {config_path}. "
                           f"Ensure the file is a valid TOML file and contains all required sections: "
                           f"{list(self._config_classes.keys())}") from e
    
    def save_config_to_toml(self, configs: Dict[str, BaseConfig], config_file: str = "config.toml") -> None:
        """Save all configurations to TOML file."""
        config_path = self.config_dir / config_file
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Organize config data by sections
        config_data = {}
        for config_name, config_obj in configs.items():
            if config_obj is not None:
                config_data[config_name] = config_obj.to_dict()
        
        try:
            with open(config_path, "wb") as f:
                tomli_w.dump(config_data, f)
            logger.info(f"Configurations saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            raise
    
    def get_config(self, config_type: str, config_file: str = "config.toml") -> BaseConfig:
        """Get specific configuration by type."""
        if config_type not in self._config_classes:
            raise ValueError(f"Unknown configuration type: {config_type}. "
                           f"Available types: {list(self._config_classes.keys())}")
        
        # Check cache first
        cache_key = f"{config_type}:{config_file}"
        if cache_key in self._configs:
            return self._configs[cache_key]
        
        # Load from file
        all_configs = self.load_config_from_toml(config_file)
        config = all_configs.get(config_type)
        
        if config is None:
            config = self._config_classes[config_type]()
        
        if config._initialize is False:
            config.initialize()

        # Cache the config
        self._configs[cache_key] = config
        return config
    
    def get_all_configs(self, config_file: str = "config.toml") -> Dict[str, BaseConfig]:
        """Get all configurations."""
        return self.load_config_from_toml(config_file)
    
    def update_config(self, config_type: str, save: bool = True, **kwargs) -> None:
        """Update specific configuration.
        
        Args:
            config_type: Type of configuration to update
            save: Whether to save changes to file
            **kwargs: Configuration values to update
        """
        config = self.get_config(config_type)
        config.update(**kwargs)
        
        if save:
            self.save_single_config(config_type, config)
    
    def save_single_config(self, config_type: str, config: BaseConfig) -> None:
        """Save a single configuration to file by merging with existing configs."""
        all_configs = self.get_all_configs()
        all_configs[config_type] = config
        self.save_config_to_toml(all_configs)
    
    def list_config_files(self) -> list[str]:
        """List all configuration files in the config directory."""
        if not self.config_dir.exists():
            return []
        
        return [f.name for f in self.config_dir.glob("*.toml")]
    
    def register_config_class(self, config_name: str, config_class: Type[BaseConfig]) -> None:
        """Register a new configuration class."""
        self._config_classes[config_name] = config_class
        logger.info(f"Registered configuration class: {config_name}")
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._configs.clear()


# Global config manager instance
config_manager = ConfigManager()

# Global configurations - loaded lazily
_global_configs: Dict[str, BaseConfig] = {}

def get_config(config_type: str = "global") -> BaseConfig:
    """Get global configuration instance by type.
    
    Args:
        config_type: Type of configuration ('global', 'mia', 'kpa', 'dosa', 'rag')
    
    Returns:
        Configuration instance
    """
    global _global_configs
    if config_type not in _global_configs:
        _global_configs[config_type] = config_manager.get_config(config_type)
    return _global_configs[config_type]

def get_global_config() -> GlobalConfig:
    """Get global configuration."""
    return get_config("global")

def get_mia_config() -> MIAConfig:
    """Get global MIA configuration."""
    return get_config("mia")

def get_kpa_config() -> KPAConfig:
    """Get global KPA configuration.""" 
    return get_config("kpa")

def get_dosa_config() -> DoSAConfig:
    """Get global DoSA configuration."""
    return get_config("dosa")

def get_rag_config() -> RAGConfig:
    """Get global RAG configuration."""
    return get_config("rag")

def get_all_configs() -> Dict[str, BaseConfig]:
    """Get all global configurations."""
    global _global_configs
    if not _global_configs:
        _global_configs = config_manager.get_all_configs()
    return _global_configs

def update_config(config_type: str = "mia", save: bool = True, **kwargs) -> None:
    """Update global configuration.
    
    Args:
        config_type: Type of configuration to update
        save: Whether to save changes to file
        **kwargs: Configuration values to update
    """
    global _global_configs
    if config_type not in _global_configs:
        _global_configs[config_type] = config_manager.get_config(config_type)
    
    _global_configs[config_type].update(**kwargs)
    
    if save:
        config_manager.save_single_config(config_type, _global_configs[config_type])

def save_config(config_type: str = None) -> None:
    """Save global configuration(s) to file.
    
    Args:
        config_type: Specific configuration type to save, or None to save all
    """
    global _global_configs
    if config_type:
        if config_type in _global_configs:
            config_manager.save_single_config(config_type, _global_configs[config_type])
    else:
        if _global_configs:
            config_manager.save_config_to_toml(_global_configs)

def save_all_configs() -> None:
    """Save all global configurations to file."""
    save_config()

def reset_config(config_type: str = None) -> None:
    """Reset configuration(s) to defaults.
    
    Args:
        config_type: Specific configuration type to reset, or None to reset all
    """
    global _global_configs
    if config_type:
        if config_type in config_manager._config_classes:
            _global_configs[config_type] = config_manager._config_classes[config_type]()
    else:
        _global_configs.clear()

def clear_config_cache() -> None:
    """Clear global configuration cache."""
    global _global_configs
    _global_configs.clear()
    config_manager.clear_cache()

# Backward compatibility functions
def update_mia_config(save: bool = True, **kwargs) -> None:
    """Update global MIA configuration."""
    update_config("mia", save=save, **kwargs)

def save_mia_config() -> None:
    """Save global MIA configuration."""
    save_config("mia")
