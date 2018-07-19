import gspread
from oauth2client.service_account import ServiceAccountCredentials
from os.path import join, dirname
from dotenv import load_dotenv
import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import datetime
from matplotlib.colors import LinearSegmentedColormap
from math import ceil
import seaborn as sns


dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

SECRET_TOKEN = os.getenv("CLIENT_SECRET_JSON")

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name(SECRET_TOKEN,
                                                         scope)
client = gspread.authorize(creds)

# Find a workbook by name and open the first sheet
# Make sure you use the right name here.
# spreadsheet = client.open("MARKET FRAGMENTATION RESEARCH_copy")

# Extract and print all of the values
# list_of_hashes = sheet.get_all_records()
# print(list_of_hashes)


def check_type(text):
    if type(text) == float:
        pass
    else:
        try:
            text = float(text.replace(',', ''))
        except ValueError:
            text = np.nan
    return text


def sheet_cols(st):
    p = r'Traffic$'  # match traffic at end of str
    cols = [s for s in list(st.keys()) if re.search(p, s)]
    return cols


class FragSpreadsheet(object):
    def __init__(self, workbook_name):
        self.worksheets = client.open(workbook_name)
        keys = [w.title for w in self.worksheets]
        self.overview = keys[0]
        self.countries_list = keys[1:]
        self.countries = None

        self._open_country()
        self._open_overview()
        self._read_overview()

    def _open_country(self):
        countries = [self.worksheets.worksheet(w) for w in self.countries_list]
        self.countries = countries

    def _open_overview(self):
        self.overview = self.worksheets.worksheet(self.overview)

    def _read_overview(self):
        records = self.overview.get_all_records()

        def format_cells(key, scale):
            return [ceil(check_type(d[key]) / scale) for d in records]

        def check_type(text):
            if type(text) == float:
                pass
            else:
                try:
                    text = float(text.replace(',', ''))
                except ValueError:
                    text = 0
            return text

        countries = [d['Country'] for d in records]
        internet_users = format_cells('Internet Users                        (31 Dec 2017)', 1.0e6)
        population = format_cells('Population              (2018 Est.) ', 1.0e6)
        fb = format_cells('Facebook Subscribers (31-Dec-2017)', 1.0e6)
        growth = format_cells('Internet Growth %               (2000 - 2017)', 1.0e3)
        data = {'internet_users': pd.Series(internet_users, index=countries),
                'population': pd.Series(population, index=countries),
                'facebook_subscribers': pd.Series(fb, index=countries),
                'growth': pd.Series(growth, index=countries)}

        self.overview = pd.DataFrame(data)

    def plot(self, growth=False, social=False):
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        if growth:
            df = self.overview.sort_values('growth')
            val = 'growth'
            side_bar_ttl = 'Internet Growth(in 1000s of %)'
            save_fig_ttl = '%s_%s.png' % ('Internet_Growth', date)
        if social:
            df = self.overview.sort_values('facebook_subscribers')
            val = 'facebook_subscribers'
            side_bar_ttl = 'FB users (in millions)'
            save_fig_ttl = '%s_%s.png' % ('FB_users', date)
        fig = plt.figure(figsize=(26, 22))
        ax = fig.add_subplot(111)
        ttl = 'Internet Usage'
        a = 0.7

        num_shades = len(df)
        customcmap = sns.cubehelix_palette(num_shades)

        df['population'].plot(kind='barh',
                              ax=ax,
                              alpha=a,
                              legend=False,
                              color=customcmap,
                              edgecolor='w',
                              xlim=(0, np.max(df['population'])),
                              title=ttl)
        ax.grid(False)
        ax.set_frame_on(False)
        ax.set_title(ax.get_title(), fontsize=26, alpha=a, ha='left')
        plt.subplots_adjust(top=0.9)
        ax.title.set_position((0, 1.08))
        ax.xaxis.set_label_position('top')
        xlab = 'Population (in millions)'
        ax.set_xlabel(xlab, fontsize=20, alpha=a, ha='left')
        ax.xaxis.set_label_coords(0, 1.04)
        ax.xaxis.tick_top()
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')

        max_v = ceil(np.max(df['population']))
        mean_v = ceil(max_v / 2.0)
        min_v = ceil(mean_v - mean_v / 2.0)
        min2_v = ceil(min_v - min_v / 2.0)

        xticks = [min2_v, min_v, mean_v, max_v]
        ax.xaxis.set_ticks(xticks)
        ax.set_xticklabels(xticks, fontsize=16, alpha=a)
        yticks = [item.get_text() for item in ax.get_yticklabels()]
        ax.set_yticklabels(yticks, fontsize=16, alpha=a)
        ax.yaxis.set_tick_params(pad=12)
        hmin, hmax = 0.1, 0.9
        xmin, xmax = np.min(df['internet_users']), np.max(df['internet_users'])
        f = lambda x: hmin + (hmax - hmin) * (x - xmin) / (xmax - xmin)
        hs = [f(x) for x in df['internet_users']]
        for container in ax.containers:
            for i, child in enumerate(container.get_children()):
                child.set_y(child.get_y() - 0.125 + 0.5 - hs[i] / 2)
                plt.setp(child, height=hs[i])
        l1 = Line2D([], [], linewidth=3, color='k', alpha=a)
        l2 = Line2D([], [], linewidth=18, color='k', alpha=a)
        l3 = Line2D([], [], linewidth=32, color='k', alpha=a)
        rnd = 1
        labels = [str(int(round(l / rnd) * rnd))
                  for l in (np.min(df['internet_users']),
                            np.mean(df['internet_users']),
                            np.max(df['internet_users']))]
        leg = ax.legend([l1, l2, l3], labels, ncol=3, frameon=False, fontsize=16,
                        bbox_to_anchor=[1.1, 0.12], handlelength=2,
                        handletextpad=1, columnspacing=2,
                        title='Internet Users (in millions)')
        plt.setp(leg.get_title(), fontsize=20, alpha=a)
        leg.get_title().set_position((0, 10))
        [plt.setp(label, alpha=a) for label in leg.get_texts()]
        ctb = LinearSegmentedColormap.from_list('custombar', customcmap, N=2048)

        sm = plt.cm.ScalarMappable(cmap=ctb,
                                   norm=mpl.colors.Normalize(vmin=df[val].min(),
                                                             vmax=df[val].max()))
        sm._A = []
        cbar = plt.colorbar(sm, alpha=0.05, aspect=10, shrink=0.4)
        cbar.solids.set_edgecolor("face")
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=16)
        smin = np.min(df[val])
        smax = np.max(df[val])
        mytks = range(smin, smax, int(smax / 10))
        cbar.set_ticks(mytks)
        cbar.ax.set_yticklabels([str(az) for az in mytks], alpha=a)
        cbar.set_label(side_bar_ttl, alpha=a, rotation=270, fontsize=20, labelpad=20)
        cbarytks = plt.getp(cbar.ax.axes, 'yticklines')
        plt.setp(cbarytks, visible=False)
        plt.savefig(save_fig_ttl, bbox_inches='tight', dpi=300)

    def plot_segmentation(self, aggregated_df):
        # get rid off SA because it skews the results
        aggregated_df = aggregated_df[aggregated_df.country != 'South africa']
        aggregated_df = aggregated_df.sort_values('country')
        fig, ax = plt.subplots(figsize=(26, 22))
        aggregated_df.set_index(['country', aggregated_df.index]).unstack()['total_visits'].plot(kind='barh',
                                                                                                 stacked=True,
                                                                                                 ax=ax,
                                                                                                 colormap="Dark2",
                                                                                                 title='Market Segmentation - Total monthly visits',
                                                                                                 legend=False)
        a = 0.7
        ax.grid(False)
        # Remove plot frame
        ax.set_frame_on(False)
        ax.set_title(ax.get_title(), fontsize=26, alpha=a)
        plt.xticks(size=20)
        plt.yticks(size=20)
        ax.yaxis.label.set_visible(False)
        ax.xaxis.set_label('Total monthly visits')
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        save_fig_ttl = '%s_%s.png' % ('Fragmentation', date)
        plt.savefig(save_fig_ttl, bbox_inches='tight', dpi=300)


class FragSheet(object):
    def __init__(self, sheet):
        self.records = sheet.get_all_records()
        self.country = sheet.title
        self.df = None

    def read(self):
        # sheet in a spreadsheet
        # record is one column in a sheet
        sites = [r['Name of Site'] for r in self.records]
        country_rank = [check_type(r['Rank in Country']) for r in self.records]
        # all columns have the same names
        cols = sheet_cols(self.records[0])
        mean_visits = []
        scale = 1000.0
        for r in self.records:
            vals = [check_type(r[c])/scale for c in cols]
            mean_visits.append(np.mean(vals))
        country = [self.country.capitalize()]*len(sites)
        data = {'total_visits': pd.Series(mean_visits, index=sites),
                'country_rank': pd.Series(country_rank, index=sites),
                'country': pd.Series(country, index=sites)}
        self.df = pd.DataFrame(data)

    def plot_country_frag(self):
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        ttl = '%s - Monthly Visits (in thousands)' % self.country.capitalize()
        a = 0.7
        ax = self.df.total_visits.plot(
            kind='barh',
            color=['#AA0000'],
            ax=ax,
            alpha=a,
            legend=False,
            edgecolor='w',
            title=ttl
        )
        # Customize title, set position, allow space on top of plot for title
        ax.set_title(ax.get_title(),
                     fontsize=26,
                     alpha=a,
                     ha='left')
        plt.subplots_adjust(top=0.9)
        ax.title.set_position((0, 1.08))

        # Set x axis label on top of plot, set label text
        # ax.xaxis.set_label_position('top')
        # xlab = 'Monthly Visits (in thousands)'
        # ax.set_xlabel(xlab, fontsize=30, alpha=a, ha='left')
        # ax.xaxis.set_label_coords(0, 1.04)

        # Position x tick labels on top
        ax.xaxis.tick_top()
        # Remove tick lines in x and y axes
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')

        # Customize x tick lables
        mean_v = np.mean(self.df['total_visits'])
        max_v = np.max(self.df['total_visits'])
        min_v = mean_v - mean_v/2.0
        xticks = [min_v, mean_v, max_v]

        ax.xaxis.set_ticks(xticks)
        ax.set_xticklabels(xticks, fontsize=16, alpha=a)

        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax.patch.set_facecolor('#FFFFFF')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['left'].set_linewidth(1)
        sites = self.df.index.values
        ax.yaxis.set_ticklabels(sites, fontstyle='italic', fontsize=16)  # ={'weight':'bold'}
        ax.set_frame_on(False)

        # Set bar height dependent on country extension
        # Set min and max bar thickness (from 0 to 1)
        hmin, hmax = 0.1, 0.9
        xmin, xmax = np.min(self.df['country_rank']), \
                     np.max(self.df['country_rank'])
        # Function that interpolates linearly between hmin and hmax
        f = lambda x: hmin + (hmax - hmin) * (x - xmin) / (xmax - xmin)
        # Make array of heights
        hs = [f(x) for x in self.df['country_rank']]

        # Iterate over bars
        for container in ax.containers:
            # Each bar has a Rectangle element as child
            for i, child in enumerate(container.get_children()):
                # Reset the lower left point of each bar so that bar is centered
                child.set_y(child.get_y() - 0.125 + 0.5 - hs[i] / 2)
                # Attribute height to each Recatangle according to country's size
                plt.setp(child, height=hs[i])

        # Legend
        # Create fake labels for legend
        l1 = Line2D([], [], linewidth=3, color='k', alpha=a)
        l2 = Line2D([], [], linewidth=18, color='k', alpha=a)
        l3 = Line2D([], [], linewidth=32, color='k', alpha=a)

        # Set three legend labels to be min, mean and max of countries extensions
        rnd = 1
        labels = [str(int(round(l / rnd) * rnd)) for l in (np.min(self.df['country_rank']),
                                                           np.mean(self.df['country_rank']),
                                                           np.max(self.df['country_rank']))]

        # Position legend in lower right part
        # Set ncol=3 for horizontally expanding legend
        leg = ax.legend([l1, l2, l3], labels, ncol=3, frameon=False, fontsize=16,
                        bbox_to_anchor=[0.9, 0.1], handlelength=2,
                        handletextpad=1, columnspacing=2, title='Rank')

        # Customize legend title
        # Set position to increase space between legend and labels
        plt.setp(leg.get_title(), fontsize=20, alpha=a)
        leg.get_title().set_position((0, 10))
        # Customize transparency for legend labels
        [plt.setp(label, alpha=a) for label in leg.get_texts()]
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        plot_name = '%s_monthly_visits_%s.png' % (self.country.capitalize(), date)
        plt.savefig(plot_name, bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == "__main__":

    frag_model = FragSpreadsheet("MARKET FRAGMENTATION RESEARCH_copy")

    aggregated = []
    for c in frag_model.countries:
        sheet = FragSheet(c)
        print(sheet.country)
        try:
            sheet.read()
            aggregated.append(sheet.df)
            # sheet.plot_country_frag()
        except (ValueError, IndexError):
            print('---> No data!')
            pass

    frag_model.plot_segmentation(pd.concat(aggregated))

    ov_df = frag_model.overview
    print(ov_df.head())
    # frag_model.plot(growth=True)
    # frag_model.plot(social=True)
