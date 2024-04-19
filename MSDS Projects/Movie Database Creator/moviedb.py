#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter


class MovieDBError(ValueError):
    """Create a subclass of the ValueError class"""
    pass


class MovieDB:
    """Create a class for creating and manipulating the movies and
    directors DataFrame.

    Attributes
    ----------
    data_dir : string
        Directory path to where the data files are located
    """
    def __init__(self, data_dir):
        """Initialize properties of class MovieDB

        Parameters
        ----------
        data_dir : string
            Directory path to where the data files are located
        """
        self.data_dir = data_dir

    def add_movie(self, title, year, genre, director):
        """Append the movie to the end of the movie.csv, if it exists,
        or create it otherwise. In particular, add the given title,
        year, genre, and the generated movie id and director id to the
        movies database. Likewise, append the director to the end of
        the directors.csv, if it exists, or create it otherwise. In
        particular, add the generated director id, and the given last
        name and given name to the directors database. If successfully
        added, return the movie id, otherwise, raise MovieDBError.

        Parameters
        ----------
        title : string
            Title of the movie
        year : integer
            Released year of the movie
        genre : string
            Genre of the movie
        director : string
            Director of the movie in 'Last name, First name' format

        Raises
        ------
        MovieDBError : ValueError
            Raised if the movie that is being added is already in the
            movie database

        Returns
        -------
        movie_id : integer
            Movie id of the added movie
        """
        title = title.strip()
        genre = genre.strip()
        given_name = director.split(',')[1].strip()
        last_name = director.split(',')[0].strip()

        # Checking if the movies file already exists
        if os.path.exists(os.path.join(self.data_dir, 'movies.csv')):
            movies_df = pd.read_csv(os.path.join(self.data_dir,
                                                 'movies.csv'))
        else:
            movies_df = pd.DataFrame(data=None, index=None,
                                     columns=['movie_id', 'title',
                                              'year', 'genre',
                                              'director_id'])

        # Checking if the directors file already exists
        if os.path.exists(os.path.join(self.data_dir,
                                       'directors.csv')):
            directors_df = pd.read_csv(os.path.join(self.data_dir,
                                                    'directors.csv'))
        else:
            directors_df = pd.DataFrame(data=None, index=None,
                                        columns=['director_id',
                                                 'given_name',
                                                 'last_name'])

        # Populating the directors csv file
        g_lower = directors_df.given_name.str.lower()
        l_lower = directors_df.last_name.str.lower()

        if (~np.any(g_lower.str.contains(given_name.lower())) or
                ~np.any(l_lower.str.contains(last_name.lower()))):
            try:
                director_id = list(directors_df.director_id)[-1] + 1
            except IndexError:
                director_id = 1
        else:
            director_id = (directors_df.loc[(g_lower ==
                                             given_name.lower()) &
                                            (l_lower ==
                                             last_name.lower()),
                                            'director_id']).iloc[0]

        if (~np.any(g_lower.str.contains(given_name.lower())) or
                ~np.any(l_lower.str.contains(last_name.lower()))):
            direct_elem = [director_id, given_name, last_name]
            direct_col = ['director_id', 'given_name', 'last_name']
            new_direct = dict(zip(direct_col, direct_elem))
            directors_df = directors_df.append(new_direct,
                                               ignore_index=True)
            directors_df.to_csv(os.path.join(self.data_dir,
                                             'directors.csv'),
                                index=False)

        # Populating the movies csv file
        t_lower = movies_df.title.str.lower()
        gr_lower = movies_df.genre.str.lower()
        y_string = movies_df.year.astype(str)
        d_string = movies_df.director_id.astype(str)
        try:
            movie_id = list(movies_df.movie_id)[-1] + 1
        except IndexError:
            movie_id = 1

        if (any([~np.any(t_lower.str.contains(title.lower())),
                 ~np.any(gr_lower.str.contains(genre.lower())),
                 ~np.any(y_string.str.contains(str(year))),
                 ~np.any(d_string.str.contains(str(director_id)))])):
            movie_elem = [movie_id, title, year, genre, director_id]
            movie_col = ['movie_id', 'title', 'year', 'genre',
                         'director_id']
            new_movie = dict(zip(movie_col, movie_elem))
            movies_df = movies_df.append(new_movie, ignore_index=True)
            movies_df.to_csv(os.path.join(self.data_dir, 'movies.csv'),
                             index=False)
            return movie_id
        else:
            raise MovieDBError

    def add_movies(self, movies):
        """Add a list of movies, in the form of dictionaries, to the
        movie database. Print `Warning: movie {title} is already in the
        database. Skipping...` if the movie is already in the database.
        Print `Warning: movie index {i} has invalid or incomplete
        information. Skipping...` if a movie has invalid or incomplete
        information.

        Parameters
        ----------
        movies : list
            List containing a dictionary with `title`, `year`, `genre`,
            and `director` as keys

        Returns
        -------
        movie_list_sort : list
            Sorted list of the movie ids added to the movie database
        """
        movie_list = []
        for i in range(len(movies)):
            title = movies[i].get('title')
            year = movies[i].get('year')
            genre = movies[i].get('genre')
            director = movies[i].get('director')

            if any(value is None for value in {title, year, genre,
                                               director}):
                print(f'Warning: movie index {i} has invalid or '
                      'incomplete information. Skipping...')
            else:
                try:
                    self.add_movie(title, year, genre, director)
                    movies_df = pd.read_csv(os.path.join(self.data_dir,
                                                         'movies.csv'))
                    movie_id = list(movies_df.movie_id)[-1]
                    movie_list.append(movie_id)
                except MovieDBError:
                    print(f'Warning: movie {title} is already in '
                          'the database. Skipping...')
                    continue
        movie_list_sort = sorted(movie_list)
        return movie_list_sort

    def delete_movie(self, movie_id):
        """Remove a movie based on the given movie id if it exists in
        the movie database, raise a MovieDBError otherwise.

        Parameters
        ----------
        movie_id : integer
            The movie id of the movie to be removed in the movie
            database

        Raises
        ------
        MovieDBError : ValueError
            Raised if the movie id is not found in the movie database
        """
        # Checking if the movies file already exists
        if os.path.exists(os.path.join(self.data_dir, 'movies.csv')):
            movies_df = pd.read_csv(os.path.join(self.data_dir,
                                                 'movies.csv'))
        else:
            movies_df = pd.DataFrame(data=None, index=None,
                                     columns=['movie_id', 'title',
                                              'year', 'genre',
                                              'director_id'])

        # Deleting a movie in the movie database
        if movie_id not in movies_df.movie_id.unique():
            raise MovieDBError
        else:
            movies_df = (movies_df.set_index('movie_id')
                         .drop(movie_id)
                         .reset_index())
            movies_df.to_csv(os.path.join(self.data_dir, 'movies.csv'),
                             index=False)

    def search_movies(
            self, title=None, year=None,
            genre=None, director_id=None):
        """Search through the movie database and return a list of
        matching movie ids based on the keyword arguments, if given,
        raise a MovieDBError otherwise.

        Parameters
        ----------
        title : string
            Title of the movie
        year : integer
            Released year of the movie
        genre : string
            Genre of the movie
        director_id : integer
            Director id of the director

        Raises
        ------
        MovieDBError : ValueError
            Raised if no nontrivial argument is passed

        Returns
        -------
        new_movies : list
            List of matching movie ids
        """
        if title == "":
            title = None
        elif genre == "":
            genre = None
        elif year == "":
            year = None
        elif director_id == "":
            director_id = None

        if all(value is None for value in [title, year, genre,
                                           director_id]):
            raise MovieDBError
        else:
            # Checking if the movies file already exists
            if os.path.exists(os.path.join(self.data_dir,
                                           'movies.csv')):
                movies_df = pd.read_csv(os.path.join(self.data_dir,
                                                     'movies.csv'))
            else:
                movies_df = pd.DataFrame(data=None, index=None,
                                         columns=['movie_id', 'title',
                                                  'year', 'genre',
                                                  'director_id'])

            # Searching the movie in the movie database
            if title is None:
                pass
            else:
                title = title.lower().strip()

            if genre is None:
                pass
            else:
                genre = genre.lower().strip()

            movie_search = (movies_df.loc[
                    (movies_df.title.str.lower() == title) |
                    (movies_df.year == year) |
                    (movies_df.genre.str.lower() == genre) |
                    (movies_df.director_id == director_id)
            ])
            new_movies = movie_search.movie_id.values.tolist()
            return new_movies

    def export_data(self):
        """Return a data frame, containing all the movies in the movie
        database, with `title`, `year`, `genre`, `director_last_name`,
        and `director_given_name` as columns.

        Returns
        -------
        df : pandas DataFrame
            Contains all the movies sorted by movie id
        """
        # Checking if the movies file already exists
        if os.path.exists(os.path.join(self.data_dir, 'movies.csv')):
            movies_df = pd.read_csv(os.path.join(self.data_dir,
                                                 'movies.csv'))
        else:
            movies_df = pd.DataFrame(data=None, index=None,
                                     columns=['movie_id', 'title',
                                              'year', 'genre',
                                              'director_id'])

        # Checking if the directors file already exists
        if os.path.exists(os.path.join(self.data_dir, 'directors.csv')):
            directors_df = pd.read_csv(os.path.join(self.data_dir,
                                                    'directors.csv'))
        else:
            directors_df = pd.DataFrame(data=None, index=None,
                                        columns=['director_id',
                                                 'given_name',
                                                 'last_name'])

        # Creating the movie data frame
        df = (movies_df.merge(directors_df, on='director_id',
                              how='left')
                       .rename(columns={
                           'last_name': 'director_last_name',
                           'given_name': 'director_given_name'})
                       .sort_values('movie_id')
                       .drop(columns=['movie_id', 'director_id'])[[
                           'title', 'year', 'genre',
                           'director_last_name',
                           'director_given_name']])
        return df

    def generate_statistics(self, stat):
        """Generate the number of movies depending on the given `stat`
        parameter.

        Parameters
        ----------
        stat : string
            The string should only be `movie`, `genre`, `director`, and
            `all`

        Raises
        ------
        MovieDBError :
            Raised if the given `stat` parameter is not `movie`,
            `genre`, `director`, or `all`

        Returns
        -------
        movie_dict : dictionary
            A dictionary with year as key and the number of movies for
            every year as value
        genre_dict : dictionary
            A dictionary with unique genre as key and another
            dictionary, with year as key and number of movies
            of that genre as value, as value
        dname_dict : dictionary
            A dictionary with director, following the format 'Last Name,
            Given Name', as key and another dictionary, with year as
            key and number of movies of that director for that year as
            value, as value
        all_ : dictionary
            A dictionary where 'movie', 'genre', and 'director' as keys
            and the corresponding dictionary per keywords as values
        """
        if stat not in ['movie', 'genre', 'director', 'all']:
            raise MovieDBError
        else:
            # Checking if the movies file already exists
            if os.path.exists(os.path.join(self.data_dir,
                                           'movies.csv')):
                movies_df = pd.read_csv(os.path.join(self.data_dir,
                                                     'movies.csv'))
            else:
                movies_df = pd.DataFrame(data=None, index=None,
                                         columns=['movie_id', 'title',
                                                  'year', 'genre',
                                                  'director_id'])

            # Checking if the directors file already exists
            if os.path.exists(os.path.join(self.data_dir,
                                           'directors.csv')):
                directors_df = pd.read_csv(os.path
                                           .join(self.data_dir,
                                                 'directors.csv'))
            else:
                directors_df = pd.DataFrame(data=None, index=None,
                                            columns=['director_id',
                                                     'given_name',
                                                     'last_name'])

            # Generating statistics per stat parameter
            df = movies_df.merge(directors_df, on='director_id',
                                 how='left')
            df['director_name'] = df.last_name + ', ' + df.given_name

            year_df = (df.groupby('year')['movie_id'].count()
                         .reset_index()
                         .rename(columns={'movie_id': 'count'}))
            movie_dict = dict(zip(year_df.year, year_df['count']))

            genre_df = {k: f.groupby('year')['movie_id'].count()
                        .to_dict() for k, f in df.groupby('genre')}
            genre_dict = {key: dict(sorted(val.items(),
                                           key=lambda x: (x[1], x[0]),
                                           reverse=True))
                          for key, val in genre_df.items()}

            dname_df = {k: f.groupby('year')['movie_id'].count()
                        .to_dict() for k, f in
                        df.groupby('director_name')}
            dname_dict = {key: dict(sorted(val.items(),
                                           key=lambda x: (x[1], x[0]),
                                           reverse=True))
                          for key, val in dname_df.items()}

            if stat == 'movie':
                return movie_dict

            elif stat == 'genre':
                return genre_dict

            elif stat == 'director':
                return dname_dict

            elif stat == 'all':
                all_attributes = ['movie', 'genre', 'director']
                all_dict = [movie_dict, genre_dict, dname_dict]
                all_ = dict(zip(all_attributes, all_dict))
                return all_
            else:
                pass

    def plot_statistics(self, stat):
        """Create a plot based on the given `stat` parameter.

        Parameters
        ----------
        stat : string
            The string should only be 'movie', 'genre', and 'director'

        Returns
        -------
        ax : matplotlib Axes
            Bar plot of the number of movies per year if the given
            `stat` parameter is `movie`.

            Line plots of the number of movies for each genre per
            year, one line per genre if the given `stat` parameter is
            `genre`.

            Line plots of the number of movies of the top 5 directors
            with the most movies per year, one director per line, if
            the given `stat` parameter is `director`.
        """
        attributes = ['movie', 'genre', 'director']
        if stat not in attributes:
            raise MovieDBError
        else:
            # Checking if the movies file already exists
            if os.path.exists(os.path.join(self.data_dir,
                                           'movies.csv')):
                movies_df = pd.read_csv(os.path.join(self.data_dir,
                                                     'movies.csv'))

            else:
                movies_df = pd.DataFrame(data=None, index=None,
                                         columns=['movie_id', 'title',
                                                  'year', 'genre',
                                                  'director_id'])

            # Checking if the directors file already exists
            if os.path.exists(os.path.join(self.data_dir,
                                           'directors.csv')):
                directors_df = pd.read_csv(os
                                           .path
                                           .join(self.data_dir,
                                                 'directors.csv'))
            else:
                directors_df = pd.DataFrame(data=None, index=None,
                                            columns=['director_id',
                                                     'given_name',
                                                     'last_name'])

        # Plotting per `stat` parameter
        if stat == 'movie':
            year_df = (movies_df.groupby('year')['movie_id'].count()
                                .reset_index()
                                .sort_values(by='year'))
            fig, ax = plt.subplots()
            ax.bar(x='year', height='movie_id', data=year_df)
            ax.set_ylabel('movies')
            ax.set_xlabel('years')
            plt.show()
            return ax

        if stat == 'genre':
            genre_df = (movies_df.groupby(['genre', 'year'])['movie_id']
                                 .count()
                                 .reset_index()
                                 .sort_values(by='genre'))
            genre_df = genre_df.pivot_table(index='year',
                                            columns='genre',
                                            values='movie_id')
            ax = genre_df.plot.line(marker='o', linestyle='-',
                                    ylabel='movies', xlabel='years')
            plt.legend()
            plt.show()
            return ax

        if stat == 'director':
            df = movies_df.merge(directors_df, on='director_id',
                                 how='left')
            df['director_name'] = df.last_name + ', ' + df.given_name
            top_5 = (df.groupby(['director_name'])['movie_id'].count()
                       .reset_index()
                       .sort_values(by=['movie_id', 'director_name'],
                                    ascending=(False, True)))[:5]
            dname = (df[df.director_name.isin(
                top_5.director_name)].groupby(
                    ['director_name', 'year'])['movie_id'].count()
                                     .reset_index()
                                     .set_index('director_name')
                                     .pivot_table(
                                         index='year',
                                         columns='director_name',
                                         values='movie_id').T
                                     .reindex(
                                         top_5.director_name.to_list())
                                     .T)
            for name in dname.columns.to_list():
                ax = (dname[~np.isnan(dname[name])][name].plot.line(
                    marker='o', linestyle='-', ylabel='movies',
                    xlabel='years'))
            plt.legend()
            plt.show()
            return ax

    def token_freq(self):
        """Count the number of times the word appeared in all of the
        movie titles in the movie database.

        Returns
        -------
        counted : dictionary
            A dictionary with token as key and the number of times the
            word appeared in all of the titles as values
        """
        # Checking if the movies file already exists
        if os.path.exists(os.path.join(self.data_dir, 'movies.csv')):
            movies_df = pd.read_csv(os.path.join(self.data_dir,
                                                 'movies.csv'))
        else:
            movies_df = pd.DataFrame(data=None, index=None,
                                     columns=['movie_id', 'title',
                                              'year', 'genre',
                                              'director_id'])

        # Counting the number of times a token appeared
        movies_df['title'] = movies_df.title.str.casefold().str.split()
        movies_list = movies_df.title.to_list()

        movies_list2 = []
        for i in movies_list:
            for j in i:
                movies_list2.append(j)
        counted = dict(Counter(movies_list2))
        return counted
