import plt_guide as pg

from metrics.summer_days.plot_su import summer_days_plot
from metrics.frost_days.plot_fd import frost_days_plot
from metrics.wet_days.plot_r10mm import wet_days_plot

if __name__ == "__main__":
    pg.setup()

    members = [
        "r1i1p1f1", 
        "r2i1p1f1", 
        "r3i1p1f1"
    ]

    for member in members:
        # pg.save(
        #     summer_days_plot(member=member), 
        #     f"./figures/data_section/climatology/summer_days_hclim_{member}.png"
        # )

        # pg.save(
        #     frost_days_plot(member=member), 
        #     f"./figures/data_section/climatology/frost_days_hclim_{member}.png"
        # )

        pg.save(
            wet_days_plot(member=member), 
            f"./figures/data_section/climatology/wet_days_hclim_{member}.png"
        )