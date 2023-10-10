library(dplyr)
library(ggplot2)
library(ggnewscale)
library(cowplot)

# Calculate the count of rows per Age and Dataset
pop_norms_count <- pop_norms %>%
  group_by(Age, Dataset) %>%
  summarise(count_rows = n(), .groups = "drop")

# Merge the count_rows data with the original dataset
pop_norms <- left_join(pop_norms, pop_norms_count, by = c("Age", "Dataset"))

# Create a custom color palette with two colors
my_palette <- c("royalblue", "darkorange")

# Create the plot for all datasets together
plot_all_datasets <- ggplot(pop_norms, aes(y = as.factor(Age), x = TMT.PRED.AVG.filtered, fill = Dataset)) +
  geom_density_ridges2(alpha = 0.25, scale = 2) +
  scale_fill_manual(values = my_palette) +
  theme(panel.background = element_rect(fill = "white")) +
  labs(x = "Temporalis Muscle Thickness, mm", y = "Age, years") +
  facet_grid(Dataset ~ ., scales = "free_y")

# Create individual plots for each dataset
dataset_plots <- lapply(unique(pop_norms$Dataset), function(dataset_name) {
  pop_norms_filtered <- filter(pop_norms, Dataset == dataset_name)
  ggplot(pop_norms_filtered, aes(y = as.factor(Age), x = TMT.PRED.AVG.filtered, fill = Dataset)) +
    geom_density_ridges2(alpha = 0.25, scale = 2) +
    scale_fill_manual(values = my_palette) +
    theme(panel.background = element_rect(fill = "white")) +
    labs(x = "Temporalis Muscle Thickness, mm", y = "Age, years") +
    ggtitle(paste("Dataset:", dataset_name))  # Add the dataset name as the plot title
})

# Combine all plots using plot_grid
final_plot <- plot_grid(plot_all_datasets, plotlist = dataset_plots, n
