import pandas as pd
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr, beta
from plotnine import *
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from adjustText import adjust_text


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def logit(p):
	epsilon = 1e-15
	return np.log((p + epsilon) / ((1 - p) + epsilon))


def tanh(x):
	return np.tanh(x)


def eharmony(x):
	return np.exp(-x)


def laplace_correction(counter, k=1):
	return counter + k


def log_corrected_counts(counter):
	return logit(counter)


def acceptability(score):
	corrected_counts = laplace_correction(score)
	log_counts = log_corrected_counts(corrected_counts)
	return log_counts


def process_and_plot(input_path, output_path, output_path_ucla):
	# Read data from file
	data = pd.read_csv(input_path, sep='\t', names=["form","tail", "ucla", "subj", "likert_rating", "tokcount", "typc", "score"])

	# Convert columns to float
	data['likert_rating'] = data['likert_rating'].astype(float)
	data['score'] = data['score'].astype(float)

	# Add the attestedness column
	def get_attestedness(typc):
		if typc == "na":
			return 'unattested'
		elif int(typc) < 10:
			return '0 < type < 10'
		else:
			return 'type > 10'

	data['attestedness'] = data['typc'].apply(get_attestedness)

	data["eharmony"] = data["ucla"].apply(eharmony)
	# subject_plot(data)

	data = data.groupby("form").agg({
	"score": "mean",
	"ucla": "mean",
	"likert_rating":"mean",
	"eharmony":"mean",
	"attestedness": "first" # take the first 'attestedness' value encountered for each form
	}).reset_index()


	# data["logscore"] = data["score"].apply(logit)
	
	# Display correlation coefficients
	# correlation_methods = ['pearson', 'spearman', 'kendall']
	# for method in correlation_methods:
	# 	s = data['score'].corr(data['likert_rating'], method=method)
	# 	print("NT: {}: {:.3f}".format(method, s))
	# 	s = data['eharmony'].corr(data['likert_rating'], method=method)
	# 	print("UCLA: {}: {:.3f}".format(method, s))

	pearsoncorr, p = pearsonr(data['score'], data['likert_rating'])
	print('Pearson correlation: %.3f' % pearsoncorr)
	print('Pearson  pvalue: %.3f' % p)

	# 
	spearmancorr, s = spearmanr(data['score'], data['likert_rating'])
	print('Spearman correlation: %.3f' % spearmancorr)
	print('Spearman pvalue: %.3f' % s)

	kendalltaucorr, k = kendalltau(data['score'], data['likert_rating'])
	print('Kendall correlation: %.3f' % kendalltaucorr)
	print('Kendall pvalue: %.3f' % k)

	# # fig = ggplot.scatterplot(data=data, x="machine_judgment", y="likert_rating")
	# title_text = fm.FontProperties(family="Times New Roman")
	# axis_text = fm.FontProperties(family="Times New Roman")
	# body_text = fm.FontProperties(family="Times New Roman")

	# # Alter size and weight of font objects
	# title_text.set_size(16)
	# axis_text.set_size(12)
	# body_text.set_size(12)
	
	# annotation_spearman = 'Spearman: %.3f' % round(spearmancorr, 3)
	
	# # Plot the averaged data with the logistic regression curve
	# averaged_plot = (ggplot(data, aes(x='score', y='likert_rating')) +
	# 				geom_point(aes(color='attestedness', shape='attestedness')) + 
	# 				scale_color_brewer(type="qual", palette="Set1") +
	# 				geom_smooth(method='lm', mapping = aes(x='score', y='likert_rating'), color = 'gray', inherit_aes=False) +
	# 				labs(x='Predicted judgment', y='Likert rating') + 
	# 				theme(legend_position=(0.34, 0.8), legend_direction='vertical', legend_title=element_blank(),
	# 					figure_size=(3,5),
	# 					axis_line_x=element_line(size=0.6, color="black"),
	# 					axis_line_y=element_line(size=0.6, color="black"),
	# 					panel_grid_major=element_blank(),
	# 					panel_grid_minor=element_blank(),
	# 					panel_border=element_blank(),
	# 					panel_background=element_blank(),
	# 					plot_title=element_text(fontproperties=title_text),
	# 					text=element_text(fontproperties=body_text),
	# 					axis_text_x=element_text(color="black"),
	# 					axis_text_y=element_text(color="black")) +
	# 				scale_y_continuous(breaks=np.arange(1, 5.01, 1), 
	# 				limits=[1, 5.1]) +
	# 				scale_x_continuous(breaks=np.arange(0, 1.005, 0.5), 
	# 				limits=[0, 1.005]) +
	# 				geom_text(aes(x=0.5, y = 1.9), family = "Times New Roman", label = annotation_spearman) 
	# 				+ geom_text(aes(label='form', color='attestedness'), size=12, position=position_nudge(x=0.1, y=0.7), show_legend=False)
	# 				# theme_bw() +
	# 		# 		theme(panel_grid_major = element_blank(), panel_grid_minor = element_blank(), legend_margin=0.5,
	# 		# legend_key=element_rect(fill='white', color='none'), figure_size=(5, 5), legend_title_align='center', text=element_text(family="Times New Roman")
	# 		#  )
	# )
	# # Draw plotnine plot, this returns a matplotlib object
	# fig = averaged_plot.draw()

	# # Get the texts from the plot
	# texts = [child for child in fig.get_children() if isinstance(child, plt.Text)]

	# # Use adjust_text to prevent text overlapping
	# adjust_text(texts, force_text=0.05, arrowprops=dict(arrowstyle="-|>", color='r', alpha=0.5))

	# # Save the plot

	# averaged_plot.save(output_path, dpi=300)

	# pearsoncorr, p = pearsonr(data['eharmony'], data['likert_rating'])
	# print('Pearson correlation: %.3f' % pearsoncorr)
	# print('Pearson  pvalue: %.3f' % p)

	# # 
	# spearmancorr, s = spearmanr(data['eharmony'], data['likert_rating'])
	# print('Spearman correlation: %.3f' % spearmancorr)
	# print('Spearman pvalue: %.3f' % s)

	# kendalltaucorr, k = kendalltau(data['eharmony'], data['likert_rating'])
	# print('Kendall correlation: %.3f' % kendalltaucorr)
	# print('Kendall pvalue: %.3f' % k)
	# annotation_spearman = 'Spearman: %.3f' % round(spearmancorr, 3)
	
	# averaged_plot_ucla = (ggplot(data, aes(x='eharmony', y='likert_rating')) +
	# 				geom_point(aes(color='attestedness', shape='attestedness')) + 
	# 				scale_color_brewer(type="qual", palette="Set1") +
	# 				geom_smooth(method='lm', mapping = aes(x='score', y='likert_rating'), color = 'gray', inherit_aes=False) +
	# 				labs(x='Predicted judgment', y='Likert rating') + 
	# 				theme(legend_position=(0.7, 0.2), legend_direction='vertical', legend_title=element_blank(),
	# 				figure_size=(3,5),
	# 				axis_line_x=element_line(size=0.6, color="black"),
	# 				axis_line_y=element_line(size=0.6, color="black"),
	# 				panel_grid_major=element_blank(),
	# 				panel_grid_minor=element_blank(),
	# 				panel_border=element_blank(),
	# 				panel_background=element_blank(),
	# 				plot_title=element_text(fontproperties=title_text),
	# 				text=element_text(fontproperties=body_text),
	# 				axis_text_x=element_text(color="black"),
	# 				axis_text_y=element_text(color="black")) +
	# 				scale_y_continuous(breaks=np.arange(1, 5.01, 1), 
	# 				limits=[1, 5.1]) +
	# 				scale_x_continuous(breaks=np.arange(0, 1.005, 0.25), 
	# 				limits=[0, 1.005]) +
	# 				geom_text(aes(x=0.5, y = 1.9), family = "Times New Roman", label = annotation_spearman
	# 				)
	# 			)
	# averaged_plot_ucla.save(output_path_ucla, dpi=300)
	return pearsoncorr, spearmancorr, kendalltaucorr

def subject_plot(data):
	# get unique subjects
	subjects = data["subj"].unique()

	for subj in subjects:
		# filter the data for the current subject
		subj_data = data[data["subj"] == subj]
		subj_data = subj_data.groupby("form").agg({
		"score": "mean",
		"ucla": "mean",
		"likert_rating":"mean",
		"eharmony":"mean",
		"attestedness": "first" # take the first 'attestedness' value encountered for each form
		}).reset_index()

		# Replace symbols in "form" column
		replace_dict = {'rz': 'ʐ', 'sz': 'ʂ', 'cz': 'tʂ', 'dzi': 'dʐ', 'si': 'ɕ', 'zi': 'ʑ', 'ni': 'ɲ', 'dz': 'dz',  'w': 'ł', 'v': 'w', 'ch':'x'}
		subj_data['form'] = subj_data['form'].replace(replace_dict, regex=True)
		
		# plot the data for the current subject
		averaged_plot = (ggplot(subj_data, aes(x='score', y='likert_rating', color='attestedness')) +
						geom_point(alpha=0.5, show_legend=True, size = 0.7) +
						labs(title='', x='Judgement', y='Likert rating') +
						geom_text(aes(label='form'), size=12, position=position_nudge(x=0.1, y=0.7), show_legend=False) +
						theme_bw() +
						theme(panel_grid_major = element_blank(), panel_grid_minor = element_blank(), legend_margin=0.5,
					legend_key=element_rect(fill='white', color='none'), figure_size=(5, 5), legend_title_align='center', text=element_text(family="Times New Roman")
				))

		# compute the correlation for the current subject
		correlation_methods = ['pearson', 'spearman', 'kendall']
		for method in correlation_methods:
			s = subj_data['score'].corr(subj_data['likert_rating'], method=method)
			print("NT: {}: {:.2f}".format(method, s))
			s = subj_data['eharmony'].corr(subj_data['likert_rating'], method=method)
			print("UCLA: {}: {:.2f}".format(method, s))
		# save the plot for the current subject
		output_path = f"result/polish/{subj}_output.png"
		ggsave(filename=output_path, plot=averaged_plot, dpi=300)
	

if __name__ == '__main__':
	input_path = "result/EnglishJudgement_2023-05-12-11-22-23.txt"
	output_path = "result/polish/correlation_plot.png"
	output_path_ucla = "result/polish/correlation_plot_ucla.png"
	process_and_plot(input_path, output_path, output_path_ucla)
