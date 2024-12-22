import os
from datetime import datetime

# ایجاد پوشه برای ذخیره نمودارها
def create_plots_dir():
    base_dir = "plots"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # افزودن تاریخ به پوشه
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(base_dir, today)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# ذخیره نمودار
def save_plot(plt, output_dir, filename):
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
