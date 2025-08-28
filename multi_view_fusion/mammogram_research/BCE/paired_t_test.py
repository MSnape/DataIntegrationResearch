from scipy import stats
import numpy as np

# Use a paired t-test to investigate whether there is a statistically significant difference in the mean 
# accuracy between the two CNN models across matched runs
# Model a is our LFFC test accuracy
# Model b is score fusion model test accuracy 
# d_i = model_b_accuracy_i - model_a_accuracy 
# Null hypothesis H_0
# The mean of the differences is 0. 
# H_0 : mu_d = 0
# ie there is no statistically significant difference between the two model's performance.

# Alternative Hypothesis H_1:
# The mean of the differences is not 0:
# H_1 : mu_d not equal 0
# There is a significant difference between the two models. 

# This is a paired t-test (runs were matched) and it is two-tailed (testing for any significant difference, 
# not just improvement)
# Accuracy values
model_a = np.array([0.70073, 0.70073, 0.722628, 0.693431, 0.70073, 0.708029])
model_b = np.array([0.7299, 0.7226, 0.7153, 0.708, 0.6861, 0.708])

# Paired t-test
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html - 
# 'Calculate the t-test on TWO RELATED samples of scores, a and b.
# This is a test for the null hypothesis that two related or repeated samples have 
# identical average (expected) values.'
t_stat, p_value = stats.ttest_rel(model_b, model_a)

# Additional stats
differences = model_b - model_a
mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)

print("Paired t-test result:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value:     {p_value:.4f}")
print(f"  Mean diff:   {mean_diff:.4f}")
print(f"  Std dev diff:{std_diff:.4f}")

if p_value < 0.05: 
    print(f"The p-value is ", "{:.4f}".format(p_value)," . The models are statistically different at the 95% confidence level.")
else:
    print(f"The p-value is ",  "{:.4f}".format(p_value), ". Therefore the p-value ≥ 0.05: The observed difference is not statistically significant. We do not have enough evidence to claim these models are different.")