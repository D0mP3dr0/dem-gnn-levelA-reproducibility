import pytorch_lightning as pl
from rich.console import Console
from rich.table import Table

class RichMetricCallback(pl.Callback):
    def __init__(self):
        self.console = Console()
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
            
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_loss = metrics.get('val_loss', 0.0)
        
        table = Table(title=f"Ep {epoch} | Val Loss: {val_loss:.4f} | SOTA CNN Monitor", show_header=True, header_style="bold magenta")
        table.add_column("Target", style="cyan", width=15)
        table.add_column("Val MAE (m)", justify="right", style="green")
        table.add_column("Tex/Ratio", justify="right", style="magenta")
        table.add_column("Miopia/Near", justify="right", style="blue")
        table.add_column("Miopia/Far", justify="right", style="red")
        table.add_column("Ratio", justify="right", style="bold red")
        table.add_column("Status", justify="right")

        mae = metrics.get('val_mae_m', metrics.get('val_mae', 0.0))
        rmse = metrics.get('val_rmse_m', metrics.get('val_rmse', 0.0))
        tex_ratio = metrics.get('texture_ratio', 0.0)
        
        tex_style = "green"
        if tex_ratio < 0.8: tex_style = "red"
        elif tex_ratio > 1.2: tex_style = "yellow"
        
        mae_near = metrics.get('mae_near_m', 0.0)
        mae_far = metrics.get('mae_far_m', 0.0)
        ratio = metrics.get('miopia_ratio', 0.0)
        
        table.add_row(
            "Elevation (dZ)",
            f"{mae:.4f} m",
            f"[{tex_style}]{tex_ratio:.2f}[/]",
            f"{mae_near:.2f} m",
            f"{mae_far:.2f} m",
            f"{ratio:.2f}x",
            "[bold green]ACTIVE[/]"
        )
        
        

        self.console.print(table)
        self.console.print(f"   📊 LR: {trainer.optimizers[0].param_groups[0]['lr']:.2e}\n")

