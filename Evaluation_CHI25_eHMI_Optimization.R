library("rstudioapi")
setwd(dirname(getActiveDocumentContext()$path))

library(devtools)
source_url("https://raw.githubusercontent.com/M-Colley/rCode/main/r_functionality.R")

library(data.table)
library(dplyr)
library(performance)
library(ggpmisc)

library(gganimate)
library(emoa) # moba redo the pareto frontier

library(scales) # Ensure this library is loaded for `pretty_breaks`
library(GGally)
library(patchwork)
library(ggsignif)
library(knitr)



dir_path <- "./eHMI-bo-participantdata"

filesQuestionnaires <- list.files(
  path = dir_path,
  recursive = TRUE,
  pattern = "QuestionnaireResponses.csv$",
  full.names = TRUE
)

filesObservations <- list.files(
  path = dir_path,
  recursive = TRUE,
  pattern = "ObservationsPerEvaluation.csv$",
  full.names = TRUE
)



# Function to extract UserID from the file path
extract_user_id <- function(file_path) {
  # Use a regular expression to find the number after "u_"
  user_id <- sub(".*u_(\\d+)/.*", "\\1", file_path)
  return(user_id)
}

main_df <- NULL
main_df <- do.call(rbind, lapply(filesQuestionnaires, function(x) {
  df <- read.delim(x, stringsAsFactors = FALSE, sep = ",", row.names = NULL)
  df$UserID <- extract_user_id(x)
  return(df)
}))


# main_df <- do.call(rbind, lapply(filesQuestionnaires, function(x) read.delim(x, stringsAsFactors = FALSE, sep = ",", row.names = NULL)))


# Trust
main_df$trust <- (main_df$Trust1 + main_df$Trust2) / 2.0

# predicatbility
main_df$predictability <- ((6 - main_df$Understanding2) + main_df$Understanding3 + main_df$Understanding1 + (6 - main_df$Understanding4)) / 4.0

# Perceived safety
main_df$perceivedSafety <- (main_df$PerceivedSafety1 + main_df$PerceivedSafety2 + main_df$PerceivedSafety3 + main_df$PerceivedSafety4) / 4.0

main_df$acceptance <- (main_df$Acceptance1 + main_df$Acceptance2) / 2.0



main_df$TimeToCross <- as.numeric(gsub(",", ".", main_df$TimeToCross))

# objectives <- c('trust', 'predictability', 'perceivedSafety',
#                 'Aesthetics1', 'TimeToCross', 'acceptance')
#
# # Calculate the Pareto front for this participant
# this would add it over all participants
# main_df <- add_pareto_emoa_column(main_df, objectives)


# should start at 1
main_df$run <- main_df$run + 1

# see how often the stopping criterion triggered
stopping <- main_df |> group_by(UserID) |> filter(run == max(run)) |> filter(run != 20)

main_df$run <- as.factor(main_df$run)
main_df$UserID <- as.factor(main_df$UserID)







# generateMoboPlot(df = main_df,x = "run", y = "MentalLoad", ytext = "Mental Demand", fillColourGroup = 1, numberSamplingSteps = 5)

main_df %>% ggplot() +
  aes(x = run, y = MentalLoad, group = 1) +
  scale_fill_see() +
  ylab("Mental Demand") +
  theme(legend.position.inside = c(0.65, 0.85)) +
  xlab("Iteration") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9, position = position_dodge(width = 0.1)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1, alpha = 0.3) +
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.1), alpha = 0.5) +
  annotate("text", x = 2.5, y = 0.5, label = "Sampling") +
  geom_segment(aes(x = 0, y = 0.75, xend = 5.2, yend = 0.75), linetype = "dashed", color = "black") +
  annotate("text", x = 13, y = 0.5, label = "Optimization") +
  geom_segment(aes(x = 5.8, y = 0.75, xend = 20, yend = 0.75), color = "black") +
  stat_poly_eq(use_label(c("eq", "R2"))) +
  stat_poly_line(fullrange = FALSE, alpha = 0.1, linetype = "dashed", linewidth = 0.5) +
  geom_vline(aes(xintercept = 5.5), linetype = "dashed", color = "black", alpha = 0.5)
ggsave("plots/bo_runs_mental.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)




main_df %>% ggplot() +
  aes(x = run, y = trust, group = 1) +
  scale_fill_see() +
  ylab("Trust") +
  theme(legend.position.inside = c(0.65, 0.85)) +
  xlab("Iteration") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9, position = position_dodge(width = 0.1)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1, alpha = 0.3) +
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.1), alpha = 0.5) +
  annotate("text", x = 2.5, y = 0.5, label = "Sampling") +
  geom_segment(aes(x = 0, y = 0.75, xend = 5.2, yend = 0.75), linetype = "dashed", color = "black") +
  annotate("text", x = 13, y = 0.5, label = "Optimization") +
  geom_segment(aes(x = 5.8, y = 0.75, xend = 20, yend = 0.75), color = "black") +
  stat_poly_eq(use_label(c("eq", "R2")), label.y = "middle") +
  stat_poly_line(fullrange = FALSE, alpha = 0.1, linetype = "dashed", linewidth = 0.5) +
  geom_vline(aes(xintercept = 5.5), linetype = "dashed", color = "black", alpha = 0.5)
ggsave("plots/bo_runs_trust.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)



main_df %>% ggplot() +
  aes(x = run, y = predictability, group = 1) +
  scale_fill_see() +
  ylab("Predictability") +
  theme(legend.position.inside = c(0.65, 0.85)) +
  xlab("Iteration") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9, position = position_dodge(width = 0.1)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1, alpha = 0.3) +
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.1), alpha = 0.5) +
  annotate("text", x = 2.5, y = 0.5, label = "Sampling") +
  geom_segment(aes(x = 0, y = 0.75, xend = 5.2, yend = 0.75), linetype = "dashed", color = "black") +
  annotate("text", x = 13, y = 0.5, label = "Optimization") +
  geom_segment(aes(x = 5.8, y = 0.75, xend = 20, yend = 0.75), color = "black") +
  stat_poly_eq(use_label(c("eq", "R2")), label.y = "middle") +
  stat_poly_line(fullrange = FALSE, alpha = 0.1, linetype = "dashed", linewidth = 0.5) +
  geom_vline(aes(xintercept = 5.5), linetype = "dashed", color = "black", alpha = 0.5)
ggsave("plots/bo_runs_predictability.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)



main_df %>% ggplot() +
  aes(x = run, y = perceivedSafety, group = 1) +
  scale_fill_see() +
  ylab("Perceived Safety") +
  theme(legend.position.inside = c(0.65, 0.85)) +
  xlab("Iteration") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9, position = position_dodge(width = 0.1)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1, alpha = 0.3) +
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.1), alpha = 0.5) +
  annotate("text", x = 2.5, y = 0.5, label = "Sampling") +
  geom_segment(aes(x = 0, y = 0.75, xend = 5.2, yend = 0.75), linetype = "dashed", color = "black") +
  annotate("text", x = 13, y = 0.5, label = "Optimization") +
  geom_segment(aes(x = 5.8, y = 0.75, xend = 20, yend = 0.75), color = "black") +
  stat_poly_eq(use_label(c("eq", "R2"))) +
  stat_poly_line(fullrange = FALSE, alpha = 0.1, linetype = "dashed", linewidth = 0.5) +
  geom_vline(aes(xintercept = 5.5), linetype = "dashed", color = "black", alpha = 0.5)
ggsave("plots/bo_runs_perceivedSafetyScore.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)










main_df %>% ggplot() +
  aes(x = run, y = Aesthetics1, group = 1) +
  scale_fill_see() +
  ylab("Aesthetics") +
  theme(legend.position.inside = c(0.65, 0.85)) +
  xlab("Iteration") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9, position = position_dodge(width = 0.1)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1, alpha = 0.3) +
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.1), alpha = 0.5) +
  annotate("text", x = 2.5, y = 0.5, label = "Sampling") +
  geom_segment(aes(x = 0, y = 0.75, xend = 5.2, yend = 0.75), linetype = "dashed", color = "black") +
  annotate("text", x = 13, y = 0.5, label = "Optimization") +
  geom_segment(aes(x = 5.8, y = 0.75, xend = 20, yend = 0.75), color = "black") +
  stat_poly_eq(use_label(c("eq", "R2")), label.y = "middle") +
  stat_poly_line(fullrange = FALSE, alpha = 0.1, linetype = "dashed", linewidth = 0.5) +
  geom_vline(aes(xintercept = 5.5), linetype = "dashed", color = "black", alpha = 0.5)
ggsave("plots/bo_runs_Aesthetics1.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)




main_df %>% ggplot() +
  aes(x = run, y = TimeToCross, group = 1) +
  scale_fill_see() +
  ylab("Time To Cross (s)") +
  theme(legend.position.inside = c(0.65, 0.85)) +
  xlab("Iteration") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9, position = position_dodge(width = 0.1)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1, alpha = 0.3) +
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.1), alpha = 0.5) +
  annotate("text", x = 2.5, y = 0.5, label = "Sampling") +
  geom_segment(aes(x = 0, y = 0.75, xend = 5.2, yend = 0.75), linetype = "dashed", color = "black") +
  annotate("text", x = 13, y = 0.5, label = "Optimization") +
  geom_segment(aes(x = 5.8, y = 0.75, xend = 20, yend = 0.75), color = "black") +
  stat_poly_eq(use_label(c("eq", "R2"))) +
  stat_poly_line(fullrange = FALSE, alpha = 0.1, linetype = "dashed", linewidth = 0.5) +
  geom_vline(aes(xintercept = 5.5), linetype = "dashed", color = "black", alpha = 0.5)
ggsave("plots/bo_runs_TimeToCross.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)



main_df %>% ggplot() +
  aes(x = run, y = acceptance, group = 1) +
  scale_color_see() +
  ylab("Acceptance") +
  theme(legend.position.inside = c(0.65, 0.85)) +
  xlab("Iteration") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9, position = position_dodge(width = 0.1)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1, alpha = 0.3) +
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.1), alpha = 0.5) +
  annotate("text", x = 2.5, y = 0.5, label = "Sampling") +
  geom_segment(aes(x = 0, y = 0.75, xend = 5.2, yend = 0.75), linetype = "dashed", color = "black") +
  annotate("text", x = 13, y = 0.5, label = "Optimization") +
  geom_segment(aes(x = 5.8, y = 0.75, xend = 20, yend = 0.75), color = "black") +
  stat_poly_eq(use_label(c("eq", "R2")), label.y = "middle") +
  stat_poly_line(fullrange = FALSE, alpha = 0.1, linetype = "dashed", linewidth = 0.5) +
  geom_vline(aes(xintercept = 5.5), linetype = "dashed", color = "black", alpha = 0.5)
ggsave("plots/bo_runs_acceptance.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)











# NOTE: User_ID 2 did not yet have the new Group_ID, added manually (and is not important for this study)

main_df <- NULL
main_df <- do.call(rbind, lapply(filesObservations, function(x) read.delim(x, stringsAsFactors = FALSE, sep = ";", row.names = NULL)))

main_df$IsPareto <- as.logical(main_df$IsPareto)
main_df$User_ID <- as.factor(main_df$User_ID)
main_df$Run <- as.factor(main_df$Run)
main_df$Phase <- as.factor(main_df$Phase)

# some logging issues in UserID 2
main_df <- subset(main_df, User_ID != "2")


# make sure that all variables are treated as numbers and have an appropriate number of decimal points
main_df[, 8:23] <- lapply(main_df[, 8:23], as.numeric)
main_df[, 8:23] <- lapply(main_df[, 8:23], function(x) round(as.numeric(x), 3))


objectives <- c(
  "Trust", "Understanding", "MentalLoad",
  "PerceivedSafety", "Aesthetics", "Acceptance", "TimeToStartCrossing"
)




# Calculate the Pareto front for this participant
# ConditionID, GroupID not needed as not altered
main_df <- main_df |>
  group_by(User_ID) |>
  mutate(PARETO_EMOA = add_pareto_emoa_column(pick(everything()), objectives = objectives)$PARETO_EMOA) |>
  ungroup()


length(levels(main_df$User_ID))

levels(main_df$User_ID)

main_df_inverted_cog <- main_df
main_df_inverted_cog$MentalLoad <- 21 - main_df_inverted_cog$MentalLoad

# cor.vars.names must be in right order
ggstatsplot::ggcorrmat(main_df_inverted_cog |> select(all_of(objectives)), cor.vars.names = c("Trust", "Predictability", "Mental Demand", "Perceived Safety", "Aesthetics", "Acceptance", "TimeToStartCrossing"), matrix.type = "lower", )
ggsave("plots/correlations_all.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)

# ATTENTION: This would look for the Pareto Front across ALL Participants
# add_pareto_emoa_column(data = main_df, objectives = objectives)
# main_df$PARETO_EMOA <- as.logical(main_df$PARETO_EMOA)




main_df %>% ggplot() +
  aes(x = Run, y = Trust, fill = Phase, colour = Phase, group = Phase) +
  scale_color_see() +
  ylab("Trust") +
  theme(legend.position.inside = c(0.85, 0.25)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9, position = position_dodge(width = 0.1)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1, alpha = 0.3) +
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)


# This is done to look at all


# # Create an empty list to store individual plots
# plot_list <- list()
#
# # Loop through each column from 8 to 23
# for (i in 8:14) {
#   col_name <- names(main_df)[i]
#
#   # Create a ggplot for each variable
#   p <- main_df %>% ggplot() +
#     aes(x = Run, y = .data[[col_name]], fill = Phase, colour = Phase, group = Phase) +
#     scale_color_see() + # Make sure you have the required palette installed
#     ylab(col_name) +
#     theme(legend.position.inside = c(0.75, 0.25)) +
#     xlab("") +
#     stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9, position = position_dodge(width = 0.1)) +
#     stat_summary(fun = mean, geom = "line", linewidth = 1, alpha = 0.3) +
#     stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
#
#   # Add the plot to the list
#   plot_list[[col_name]] <- p
# }
#
# # Combine all plots into one big plot using patchwork or cowplot
# final_plot <- wrap_plots(plot_list, ncol = 4) # Adjust ncol to fit your needs
#
# # Print the final plot
# print(final_plot)








main_df_true <- subset(main_df, PARETO_EMOA == "TRUE")


main_df_true_inverted_cog <- main_df_true
main_df_true_inverted_cog$MentalLoad <- 21 - main_df_true_inverted_cog$MentalLoad


# cor.vars.names must be in right order
ggstatsplot::ggcorrmat(main_df_true_inverted_cog |> select(all_of(objectives)), cor.vars.names = c("Trust", "Predictability", "Mental Demand", "Perceived Safety", "Aesthetics", "Acceptance", "TimeToStartCrossing"), matrix.type = "lower", )
ggsave("plots/correlations_mobo_true.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)





# Generate the parallel coordinates plot
ggparcoord(
  data = main_df,
  columns = 8:14, # Indices of the QEHVI columns
  groupColumn = "PARETO_EMOA", # Color lines by the 'Run' column (now a factor)
  scale = "globalminmax", # Scale the columns to [0, 1] based on global min/max
  alphaLines = 0.2, # Transparency of the lines
  showPoints = TRUE, # Show points on the lines
  title = "Parallel Coordinates Plot Highlighting Pareto Frontier"
) +
  scale_color_see(reverse = TRUE) + # Generate unique colors
  theme_minimal(base_size = 15) + # Use a clean theme
  labs(color = "PARETO_EMOA") # Label the legend






main_df_true$Red <- round(main_df_true$r * 255, digits = 0)
main_df_true$Green <- round(main_df_true$g * 255, digits = 0)
main_df_true$Blue <- round(main_df_true$b * 255, digits = 0)



long_df_parameters <- main_df_true %>%
  pivot_longer(cols = 15:23, names_to = "variable", values_to = "value")


# long_df_parameters$variable <- factor(long_df_parameters$variable, levels = c("SemanticSegmentation", "SemanticSegmentationAlpha", "PedestrianIntention", "PedestrianIntentionSize", "Trajectory", "TrajectoryAlpha", "TrajectorySize", "EgoTrajectory", "EgoTrajectoryAlpha", "EgoTrajectorySize", "CoveredArea", "CoveredAreaAlpha", "CoveredAreaSize", "OccludedCars", "CarStatus", "CarStatusAlpha"))
long_df_parameters$variable <- factor(long_df_parameters$variable)
levels(long_df_parameters$variable)

# Replace the level 'a' with 'alpha'
levels(long_df_parameters$variable)[levels(long_df_parameters$variable) == "a"] <- "Alpha"
levels(long_df_parameters$variable)[levels(long_df_parameters$variable) == "b"] <- "Blue"
levels(long_df_parameters$variable)[levels(long_df_parameters$variable) == "g"] <- "Green"
levels(long_df_parameters$variable)[levels(long_df_parameters$variable) == "r"] <- "Red"
levels(long_df_parameters$variable)[levels(long_df_parameters$variable) == "blinkFrequency"] <- "Blink Frequency"
levels(long_df_parameters$variable)[levels(long_df_parameters$variable) == "horizontalWidth"] <- "Width"
levels(long_df_parameters$variable)[levels(long_df_parameters$variable) == "verticalPosition"] <- "Vertical\nPosition"
levels(long_df_parameters$variable)[levels(long_df_parameters$variable) == "verticalWidth"] <- "Height"
levels(long_df_parameters$variable)[levels(long_df_parameters$variable) == "volume"] <- "Volume"


# Relevel the factors to the desired order
long_df_parameters$variable <- factor(
  long_df_parameters$variable,
  levels = c(
    "Red", "Green", "Blue", "Alpha",
    "Blink Frequency", "Horizontal\nWidth", "Vertical\nPosition", "Vertical\nWidth", "Volume"
  )
)

# Check the new levels to confirm the reordering
levels(long_df_parameters$variable)


long_df_parameters <- long_df_parameters %>%
  select(-IsPareto)


result2 <- long_df_parameters %>%
  dplyr::group_by(variable) %>%
  dplyr::summarize(
    Q1 = quantile(value, 0.25, na.rm = TRUE),
    Q3 = quantile(value, 0.75, na.rm = TRUE)
  ) %>%
  dplyr::left_join(long_df_parameters, by = c("variable")) %>%
  dplyr::group_by(variable) %>%
  dplyr::mutate(
    dist_to_Q1 = abs(value - Q1),
    dist_to_Q3 = abs(value - Q3)
  ) %>%
  dplyr::summarize(
    Q1 = first(Q1),  # Retain Q1
    Q3 = first(Q3),  # Retain Q3
    closest_to_Q1_value = value[which.min(dist_to_Q1)],  # Value closest to Q1
    closest_to_Q1_userid = User_ID[which.min(dist_to_Q1)],  # User ID closest to Q1
    closest_to_Q3_value = value[which.min(dist_to_Q3)],  # Value closest to Q3
    closest_to_Q3_userid = User_ID[which.min(dist_to_Q3)]   # User ID closest to Q3
  )


# Convert your summarized result to Markdown format using kable
md_table <- kable(result2, format = "markdown")

# Write the Markdown table to a file
writeLines(md_table, "summary_results_iqr_ehmi.md")



# Assuming test_params is your data frame
# Group by the variable and calculate the statistics
grouped_data <- long_df_parameters %>% group_by(variable)
stats <- grouped_data %>%
  summarise(
    mean = mean(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE)
  ) %>%
  mutate(ymin = mean - 1 * sd, ymax = mean + 1 * sd)

print(stats)

# Create an offset for xmin and xmax
offset <- 0.2 # Adjust this value as needed for visibility
stats$xmin <- as.numeric(factor(stats$variable)) - offset
stats$xmax <- as.numeric(factor(stats$variable)) + offset

levels(long_df_parameters$variable)



long_df_parameters %>% ggplot() +
  aes(x = variable, y = value, group = 1) +
  geom_rect(
    data = stats, inherit.aes = FALSE,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    fill = "grey", alpha = 0.2
  ) +
  # Adding vertical line between Vertical Width and Volume
  geom_vline(xintercept = 8.5, linetype = "dashed", color = "black", size = 0.3) +
  theme(legend.position.inside = c(0.70, 0.09)) +
  xlab("Visualization Parameter") +
  ylab("Range of final and normalized value\nper visualization parameter") +
  ylim(0, 1) +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9, position = position_dodge(width = 0.1)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1, alpha = 0.1) +
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.1), alpha = 0.5) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "blue")
ggsave("plots/parameters_pareto_true_grey_rect_1_sd.pdf", width = 18, height = 10, device = cairo_pdf)






# we take Trust, Mental Demand, and TimeToStartCrossing as little correlation





#
# # Extract the necessary columns for plotting
# data_plot <- main_df[, c('Trust', 'MentalLoad', 'TimeToStartCrossing', 'PARETO_EMOA')]
#
# # Convert to numeric if not already
# data_plot$Trust <- as.numeric(data_plot$Trust)
# data_plot$MentalLoad <- as.numeric(data_plot$MentalLoad)
# data_plot$TimeToStartCrossing <- as.numeric(data_plot$TimeToStartCrossing)
#
#
#
# # Get the list of unique User_IDs
# unique_users <- unique(main_df$User_ID)
#
# # Loop through each User_ID
# for (user_id in unique_users) {
#   data_user <- subset(main_df, User_ID == user_id)
#   print(head(data_user))
#
#   # Extract the necessary columns for plotting (Trust, MentalLoad)
#   X <- as.matrix(data_user[, c('Trust', 'MentalLoad')])
#
#   # Identify Pareto-optimal points for this user
#   Xnd <- X[data_user$PARETO_EMOA == TRUE, ]
#
#   # Plot all points
#   plot(X, col = 'grey', pch = 20, xlab = 'Trust', ylab = 'Mental Load',
#        main = paste('Pareto Front for User', user_id))
#
#   # Highlight Pareto-optimal points
#   points(Xnd, col = 'red', pch = 19)
#
#   # Optionally, draw the Pareto front for this User_ID
#   plotParetoEmp(Xnd, col = 'blue', add = TRUE)
# }
#
#
# # Loop through each User_ID
# for (user_id in unique_users) {
#   # Subset data for the current User_ID
#   data_user <- subset(main_df, User_ID == user_id)
#
#   # Extract the necessary columns for plotting (Trust, MentalLoad, TimeToStartCrossing)
#   X3D <- as.matrix(data_user[, c('Trust', 'MentalLoad', 'TimeToStartCrossing')])
#
#   # Identify Pareto-optimal points for this user
#   X3D_nd <- X3D[data_user$PARETO_EMOA == TRUE, ]
#
#   # Plot all points in 3D for this user
#   open3d()  # Start a new 3D plotting window
#   plot3d(X3D, col = 'grey', size = 3,
#          xlab = 'Trust', ylab = 'Mental Load', zlab = 'Time To Start Crossing',
#          main = paste('Pareto Front for User', user_id))
#
#   # Highlight Pareto-optimal points in 3D
#   points3d(X3D_nd, col = 'red', size = 5)
#
#   # Optionally, adjust plot bounds
#   X.range <- apply(X3D, 2, range)
#   bounds <- rbind(X.range[1, ] - 0.1 * diff(X.range), X.range[2, ] + 0.1 * diff(X.range))
#
#   # Optionally, plot Pareto surface or additional elements if necessary
#   plotParetoEmp(nondominatedPoints = X3D_nd, add = TRUE, bounds = bounds, alpha = 0.5)
#
#   # Add a legend
#   legend3d("topright", legend = c("All Points", "Pareto Front"), col = c("grey", "red"), pch = 19, cex = 1)
#
#   # Pause for a moment to allow visualization (optional)
#   readline(prompt="Press [enter] to continue to the next user...")  # This will wait for user input before continuing
# }
#






#### Compare male with female ###
generate_plots <- function(main_df, response_var, label, bf_object, min, max) {
  # Extract the Bayes Factor from the ttestBF object
  bf_value <- exp(bf_object@bayesFactor$bf)

  # Conditionally format the BF annotation
  bf_annotation <- ifelse(bf_value > 100, "BF > 100", paste("BF =", round(bf_value, 2)))

  # Add a small margin to the max value to ensure the BF annotation is visible
  max_with_margin <- max + (max - min) * 0.1 # Increase the margin slightly

  # Calculate the y-position for the BF annotation
  y_position_bf <- max_with_margin - (max_with_margin - max) * 0.05 # Slightly below the upper limit

  # Plot main effect of gender
  plot <- ggplot(main_df, aes(x = gender, y = !!sym(response_var), fill = gender)) +
    theme_minimal(base_size = 15) + # Use theme_minimal for a clean look
    ylab(label) +
    theme(
      legend.position = "none", # Remove legend
      axis.title = element_text(size = 22, color = "black"),
      axis.text = element_text(size = 18, color = "black"),
      plot.title = element_text(size = 28, color = "black"),
      plot.subtitle = element_text(size = 18, color = "black"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      axis.text.x = element_text(angle = 70, hjust = 1, color = "black", size = 17) # Rotate x-axis labels
    ) +
    xlab("") +
    geom_boxplot(width = 0.3, show.legend = FALSE, aes(fill = gender)) + # Slimmer boxplots and no legend
    geom_jitter(width = 0.1, height = 0.05, color = "black", alpha = 0.2, size = 2, shape = 16) + # Remove contour of scatter plot points
    scale_fill_see() + # Specify individual hex codes
    stat_summary(fun = mean, geom = "point", shape = 21, size = 5, color = "black", fill = "white") +
    geom_signif(
      comparisons = list(c("female", "male")),
      annotations = "",
      map_signif_level = FALSE,
      textsize = 6,
      tip_length = 0.03, # Adjust tip length to make the bracket visible
      y_position = y_position_bf + (max_with_margin - max) * 0.05
    ) + # Adjust y-position of the bracket
    annotate("text", x = 1.5, y = y_position_bf, label = bf_annotation, size = 6, color = "black") + # Manually annotate BF
    ylim(min, max_with_margin) # Set the y-axis limits with margin

  # Show the plot and save as PDF
  if (!is.null(plot)) {
    print(plot)
    # Save the plot as PDF
    ggsave(filename = paste0("plots/", response_var, "_newPareto_plot.pdf"), plot = plot, width = 2.5, height = 7, device = cairo_pdf)
  }
}


# GENDER: 1 = female; 2 = male; 3 = non-binary; 4 = do not tell


demo_df <- read.csv2(file = "ehmi-optimization-chi25-demographic-data.csv")

demo_df$gender <- ifelse(demo_df$gender == 1, "female",
  ifelse(demo_df$gender == 2, "male",
    ifelse(demo_df$gender == 3, "nb", demo_df$gender)
  )
)


report::report_participants(data = demo_df)

# A3 = High school; A4 = College
table(demo_df$education)

# A2 = Student (college); A3 = Employee
table(demo_df$job)


mean(demo_df$interest.ease.)
sd(demo_df$interest.ease.)

mean(demo_df$interest.interest.)
sd(demo_df$interest.interest.)

mean(demo_df$interest.reality.)
sd(demo_df$interest.reality.)



# The final design matches my expectation.
mean(demo_df$add.UserExp.)
sd(demo_df$add.UserExp.)

# I'm pleased with the final design.
mean(demo_df$add.Satisfaction.)
sd(demo_df$add.Satisfaction.)

# I felt in control of the design process.
mean(demo_df$add.Agency.)
sd(demo_df$add.Agency.)

# I believe the design is optimal for me.
mean(demo_df$add.Confidence.)
sd(demo_df$add.Confidence.)

# I feel the final design is mine.
mean(demo_df$add.Ownership.)
sd(demo_df$add.Ownership.)










# Merging demo_df and main_df by User_ID from main_df and UserID from demo_df
merged_df <- merge(main_df, demo_df[, c("UserID", "gender")], by.x = "User_ID", by.y = "UserID")


merged_df <- subset(merged_df, gender != "nb")






merged_df |>
  group_by(gender) |>
  summarise(mean = mean(Trust), sd = sd(Trust))
merged_df |>
  group_by(gender) |>
  summarise(mean = mean(Understanding), sd = sd(Understanding))
merged_df |>
  group_by(gender) |>
  summarise(mean = mean(MentalLoad), sd = sd(MentalLoad))
merged_df |>
  group_by(gender) |>
  summarise(mean = mean(PerceivedSafety), sd = sd(PerceivedSafety))
merged_df |>
  group_by(gender) |>
  summarise(mean = mean(Acceptance), sd = sd(Acceptance))
merged_df |>
  group_by(gender) |>
  summarise(mean = mean(Aesthetics), sd = sd(Aesthetics))
merged_df |>
  group_by(gender) |>
  summarise(mean = mean(TimeToStartCrossing), sd = sd(TimeToStartCrossing))


merged_df <- subset(merged_df, PARETO_EMOA == TRUE)

merged_df |>
  group_by(gender) |>
  summarise(count_true_pareto = sum(PARETO_EMOA == TRUE, na.rm = TRUE))



# Trust
bf_trust <- ttestBF(x = merged_df$Trust[merged_df$gender == "female"], y = merged_df$Trust[merged_df$gender == "male"], paired = FALSE)
bf_trust
report::report(bf_trust)
generate_plots(merged_df, response_var = "Trust", label = "Trust in Automation", bf_object = bf_trust, min = 1, max = 5)

# Understanding
bf_understanding <- ttestBF(x = merged_df$Understanding[merged_df$gender == "female"], y = merged_df$Understanding[merged_df$gender == "male"], paired = FALSE)
bf_understanding
report::report(bf_understanding)
generate_plots(merged_df, response_var = "Understanding", label = "Predictability", bf_object = bf_understanding, min = 1, max = 5)

# Mental Demand
bf_md <- ttestBF(x = merged_df$MentalLoad[merged_df$gender == "female"], y = merged_df$MentalLoad[merged_df$gender == "male"], paired = FALSE)
bf_md
report::report(bf_md)
generate_plots(merged_df, response_var = "MentalLoad", label = "Mental Demand", bf_object = bf_md, min = 1, max = 20)


# Perceived Safety
bf_ps <- ttestBF(x = merged_df$PerceivedSafety[merged_df$gender == "female"], y = merged_df$PerceivedSafety[merged_df$gender == "male"], paired = FALSE)
bf_ps
report::report(bf_ps)
generate_plots(merged_df, response_var = "PerceivedSafety", label = "Perceived Safety", bf_object = bf_ps, min = -3, max = 3)

# Acceptance
bf_acc <- ttestBF(x = merged_df$Acceptance[merged_df$gender == "female"], y = merged_df$Acceptance[merged_df$gender == "male"], paired = FALSE)
bf_acc
report::report(bf_acc)
generate_plots(merged_df, response_var = "Acceptance", label = "Acceptance", bf_object = bf_acc, min = 1, max = 7)

# Aesthetics
bf_aes <- ttestBF(x = merged_df$Aesthetics[merged_df$gender == "female"], y = merged_df$Aesthetics[merged_df$gender == "male"], paired = FALSE)
bf_aes
report::report(bf_aes)
generate_plots(merged_df, response_var = "Aesthetics", label = "Aesthetics", bf_object = bf_aes, min = 1, max = 7)


# TimeToStartCrossing
bf_ttsc <- ttestBF(x = merged_df$TimeToStartCrossing[merged_df$gender == "female"], y = merged_df$TimeToStartCrossing[merged_df$gender == "male"], paired = FALSE)
bf_ttsc
report::report(bf_ttsc)
generate_plots(merged_df, response_var = "TimeToStartCrossing", label = "Time To Start Crossing (s)", bf_object = bf_ttsc, min = 0, max = 20)











combined_plot <- NULL

design_parameters <- c(
  "a", "b", "g", "r", "blinkFrequency", "horizontalWidth", "verticalPosition",
  "verticalWidth", "volume"
)



# Calculate the Mean and SD for each design parameter by Group
design_summary <- merged_df %>%
  group_by(gender) %>%
  summarise(across(all_of(design_parameters),
    list(
      mean = ~ mean(., na.rm = TRUE),
      sd = ~ sd(., na.rm = TRUE),
      #iqr = ~ IQR(., na.rm = TRUE),
      loweriqr = ~ quantile(., 0.25, na.rm = TRUE),  # Q1
      upperiqr = ~ quantile(., 0.75, na.rm = TRUE)   # Q3
    ),
    .names = "{.col}_{.fn}"
  )) %>%
  pivot_longer(
    cols = -gender,
    names_to = c("Parameter", ".value"),
    names_sep = "_"
  )

# Set the factor levels for the Parameter column to ensure correct order
design_summary$Parameter <- factor(design_summary$Parameter, levels = design_parameters)
print(design_summary, n = 18)







# design_summary$gender

# Calculate the BF for each design parameter and add it to the plot
bf_annotations <- list()
for (param in design_parameters) {
  # Calculate the BF for the current parameter
  bf_object <- ttestBF(
    x = merged_df[[param]][merged_df$gender == "female"],
    y = merged_df[[param]][merged_df$gender == "male"],
    paired = FALSE
  )
  bf_value <- exp(bf_object@bayesFactor$bf)

  # Calculate the mean and SD for each group
  mean_female <- mean(merged_df[[param]][merged_df$gender == "female"], na.rm = TRUE)
  sd_female <- sd(merged_df[[param]][merged_df$gender == "female"], na.rm = TRUE)

  mean_male <- mean(merged_df[[param]][merged_df$gender == "male"], na.rm = TRUE)
  sd_male <- sd(merged_df[[param]][merged_df$gender == "male"], na.rm = TRUE)

  # Print the ttestBF results and the means and SDs to the console
  cat("\nResults for parameter:", param, "\n")
  print(bf_object)
  cat("Mean (female):", round(mean_female, 2), "SD (female):", round(sd_female, 2), "\n")
  cat("Mean (male):", round(mean_male, 2), "SD (male):", round(sd_male, 2), "\n")
  
  
  # Calculate the IQR for each group
  iqr_female <- IQR(merged_df[[param]][merged_df$gender == "female"], na.rm = TRUE)
  iqr_male <- IQR(merged_df[[param]][merged_df$gender == "male"], na.rm = TRUE)
  
  # Print the IQRs
  cat("IQR (female):", round(iqr_female, 2), "\n")
  cat("IQR (male):", round(iqr_male, 2), "\n")
  

  # Conditionally format the BF annotation based on value
  if (bf_value < 0.01) {
    bf_annotation <- "<<<"
  } else if (bf_value < 0.1) {
    bf_annotation <- "<<"
  } else if (bf_value < 0.3) {
    bf_annotation <- "<"
  } else if (bf_value >= 0.3 && bf_value <= 3) {
    bf_annotation <- "="
  } else if (bf_value > 3 && bf_value <= 10) {
    bf_annotation <- ">"
  } else if (bf_value > 10 && bf_value <= 100) {
    bf_annotation <- ">>"
  } else if (bf_value > 100) {
    bf_annotation <- ">>>"
  }

  # Add the BF annotation to the list
  bf_annotations[[param]] <- bf_annotation
}

combined_plot <- NULL

# Generate the plot with dots for the mean and error bars for the SD
combined_plot <- ggplot(design_summary, aes(x = Parameter, y = mean, color = gender)) +
  #geom_point(position = position_dodge(width = 0.6), size = 3) + # Add dots for the mean
  geom_errorbar(aes(ymin = loweriqr, ymax = upperiqr),
    position = position_dodge(width = 0.6), width = 0.2
  ) + # Add error bars for SD
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", linewidth = 1) + # Add a horizontal line at y = 0.5

  theme_minimal(base_size = 15) +
  scale_color_see() + # Customize point colors
  labs(x = "Design Parameters", y = "Values (IQR)", color = "") +
  theme(
    axis.text.x = element_text(angle = 70, hjust = 1, color = "black", size = 14), # Rotate x-axis labels
    axis.title = element_text(size = 18),
    plot.title = element_text(size = 20),
    legend.position.inside = c(0.92, 0.15), # Position the legend inside the plot area
    # legend.background = element_rect(fill = "white", color = NA),  # Optional: white background for legend
    # legend.key = element_rect(fill = "white", color = NA)  # Optional: white background for legend keys
  ) +
  coord_cartesian(ylim = c(0, NA)) # Adjust y-axis to start from 0



# Add BF annotations above each parameter
combined_plot <- combined_plot +
  geom_text(
    data = design_summary %>% filter(gender == "female"),
    aes(x = Parameter, y = max(upperiqr) + 0.1 * max(upperiqr), label = bf_annotations[Parameter]),
    color = "black", size = 4, hjust = 0.5
  )

combined_plot
combined_plot <- NULL

combined_plot <- ggplot(design_summary, aes(x = Parameter, y = mean, color = gender)) +
  #geom_point(position = position_dodge(width = 0.6), size = 3) + # Add dots for the mean
  geom_errorbar(aes(ymin = loweriqr, ymax = upperiqr),
    position = position_dodge(width = 0.6), width = 0.2
  ) + 
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", size = 1) + # Add a horizontal line at y = 0.5

  # Loop over each design parameter to add jittered points for individual data
  geom_jitter(
    data = merged_df, aes(x = "a", y = a, color = gender),
    position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.6),
    size = 0.7, alpha = 0.1, inherit.aes = FALSE
  ) + # Jitter individual points
  geom_jitter(
    data = merged_df, aes(x = "b", y = b, color = gender),
    position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.6),
    size = 0.7, alpha = 0.1, inherit.aes = FALSE
  ) +
  geom_jitter(
    data = merged_df, aes(x = "g", y = g, color = gender),
    position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.6),
    size = 0.7, alpha = 0.1, inherit.aes = FALSE
  ) +
  geom_jitter(
    data = merged_df, aes(x = "r", y = r, color = gender),
    position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.6),
    size = 0.7, alpha = 0.1, inherit.aes = FALSE
  ) +
  geom_jitter(
    data = merged_df, aes(x = "blinkFrequency", y = blinkFrequency, color = gender),
    position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.6),
    size = 0.7, alpha = 0.1, inherit.aes = FALSE
  ) +
  geom_jitter(
    data = merged_df, aes(x = "horizontalWidth", y = horizontalWidth, color = gender),
    position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.6),
    size = 0.7, alpha = 0.1, inherit.aes = FALSE
  ) +
  geom_jitter(
    data = merged_df, aes(x = "verticalPosition", y = verticalPosition, color = gender),
    position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.6),
    size = 0.7, alpha = 0.1, inherit.aes = FALSE
  ) +
  geom_jitter(
    data = merged_df, aes(x = "verticalWidth", y = verticalWidth, color = gender),
    position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.6),
    size = 0.7, alpha = 0.1, inherit.aes = FALSE
  ) +
  geom_jitter(
    data = merged_df, aes(x = "volume", y = volume, color = gender),
    position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.6),
    size = 0.7, alpha = 0.1, inherit.aes = FALSE
  ) +
  theme_minimal(base_size = 15) +
  scale_color_see() + # Customize point colors
  labs(x = "Design Parameters", y = "Values (IQR)", color = "") +
  theme(
    axis.text.x = element_text(angle = 70, hjust = 1, color = "black", size = 14), # Rotate x-axis labels
    axis.title = element_text(size = 18),
    plot.title = element_text(size = 20),
    legend.position = c(0.92, 0.15) # Position the legend inside the plot area
  ) +
  scale_x_discrete(
    labels = c(
      "a" = "Alpha",
      "b" = "Blue",
      "g" = "Green",
      "r" = "Red",
      "blinkFrequency" = "Blink\nFrequency",
      "horizontalWidth" = "Width",
      "verticalPosition" = "Vertical\nPosition",
      "verticalWidth" = "Height",
      "volume" = "Volume"
    )
  ) +
  coord_cartesian(ylim = c(0, NA)) # Adjust y-axis to start from 0


combined_plot

# values = c("female" = "#0000FF", "male" = "#008000")

# Add BF annotations above each parameter
combined_plot <- combined_plot +
  geom_text(
    data = design_summary %>% filter(gender == "female"),
    aes(x = Parameter, y = max(upperiqr) + 0.1 * max(upperiqr), label = bf_annotations[Parameter]),
    color = "black", size = 4, hjust = 0.5
  )


# Print and save the combined plot
print(combined_plot)
ggsave(filename = paste0("plots/newPareto_combined_plot.pdf"), plot = combined_plot, width = 11, height = 6, device = cairo_pdf)

print(bf_annotations)
















# Example dataframe with more RGB values (replace with your actual data)
set.seed(123)
# Assuming your dataframe has 'Red', 'Green', 'Blue', and 'a' columns
df <- main_df_true %>%
  mutate(
    Red = round(r * 255, digits = 0),
    Green = round(g * 255, digits = 0),
    Blue = round(b * 255, digits = 0),
    Alpha = a
  ) # 'a' is already the alpha channel, ranging from 0 to 1

# Create a hex color column from RGB values and alpha
df <- df %>%
  mutate(color = rgb(Red / 255, Green / 255, Blue / 255, alpha = Alpha))

# Add row numbers to identify each cell
df <- df %>%
  mutate(row = row_number())

# Set the number of columns in the grid
num_cols <- 25 # Adjust this to fit your desired grid size
num_rows <- ceiling(nrow(df) / num_cols)

# Create row and column indices for the grid
df <- df %>%
  mutate(
    grid_row = (row - 1) %/% num_cols + 1,
    grid_col = (row - 1) %% num_cols + 1
  )

# Create the plot
ggplot(df, aes(x = grid_col, y = grid_row, fill = color)) +
  geom_tile(color = "white", size = 0.5) + # Create tiles with white border
  scale_fill_identity() + # Use the color column directly as fill
  theme_void() + # Remove all background elements
  theme(aspect.ratio = num_rows / num_cols) # Adjust aspect ratio based on grid














# Assuming 'User_ID' is present in the dataframe

# Assuming 'User_ID' is present in the dataframe
df <- main_df_true %>%
  mutate(
    Red = round(r * 255, digits = 0),
    Green = round(g * 255, digits = 0),
    Blue = round(b * 255, digits = 0),
    Alpha = a
  ) %>%
  mutate(color = rgb(Red / 255, Green / 255, Blue / 255, alpha = Alpha)) %>%
  group_by(User_ID) %>%
  mutate(row = row_number()) %>% # Create row number within each User_ID
  ungroup()

# Define the number of rows and columns
num_cols <- n_distinct(df$User_ID) # One column per User_ID
num_rows <- max(table(df$User_ID)) # Max number of rows within a User_ID group

# Assign grid row and column
df <- df %>%
  mutate(
    grid_col = as.numeric(factor(User_ID)), # Column corresponds to User_ID
    grid_row = row
  ) # Row corresponds to the position within the User_ID group


df$Phase <- as.factor(df$Phase)
df$Phase <- relevel(df$Phase, ref = "sampling")

phase_labels <- c(
  "optimization" = "MOBO~phase:~bold(optimization)",
  "sampling" = "MOBO~phase:~bold(sampling)"
)

# Plot the data
ggplot(df, aes(x = User_ID, y = grid_row, fill = color)) +
  geom_tile(color = "white", size = 0.5) + # Create tiles with white border
  scale_fill_identity() + # Use the color column directly as fill

  # Add row numbers as text in each tile
  geom_text(aes(label = row), color = "black", size = 2, alpha = 0.4) +

  # Add labels for x and y axes
  labs(x = "Participant", y = "Color on the Pareto front") +
  # Remove axis lines and ticks
  theme_minimal() + # Remove all background elements
  # Adjust aspect ratio based on grid size
  theme(aspect.ratio = num_rows / num_cols, axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank(), strip.text = element_text(size = 10)) +
  # Expand space below the grid for User_ID labels
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.05))) +
  facet_wrap(~ Phase, ncol = 1, labeller = as_labeller(phase_labels, label_parsed))  # Create separate plots for each 'Phase'
ggsave("plots/color_distribution.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)

