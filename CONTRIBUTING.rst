Contributing
============

pydae is a community project, hence all contributions are more than
welcome!

Bug reporting
-------------

Not only things break all the time, but also different people have different
use cases for the project. If you find anything that doesn't work as expected
or have suggestions, please refer to the `issue tracker`_ on GitHub.

.. _`issue tracker`: https://github.com/pydae/pydae/issues

Documentation
-------------

Documentation can always be improved and made easier to understand for
newcomers. The docs are stored in text files under the `docs/source`
directory, so if you think anything can be improved there please edit the
files and proceed in the same way as with `code writing`_.

The Python classes and methods also feature inline docs: if you detect
any inconsistency or opportunity for improvement, you can edit those too.

Besides, the `wiki`_ is open for everybody to edit, so feel free to add
new content.

To build the docs, you must first create a development environment (see
below) and then in the ``docs/`` directory run::

    $ cd docs
    $ make html

After this, the new docs will be inside ``build/html``. You can open
them by running an HTTP server::

    $ cd build/html
    $ python -m http.server
    Serving HTTP on 0.0.0.0 port 8000 ...

And point your browser to http://0.0.0.0:8000.

