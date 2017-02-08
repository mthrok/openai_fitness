"""WSGI compliant server module"""
from __future__ import absolute_import

__all__ = ['create_server']


def create_server(app, port=5000, host='0.0.0.0'):
    """Mount application on cherrypy WSGI server

    Parameters
    ----------
    app : a dict or list of (path_prefix, app) pairs
        See :py:func:`create_app`

    Returns
    -------
    WSGIServer object
    """
    import cheroot.wsgi
    dispatcher = cheroot.wsgi.WSGIPathInfoDispatcher({'/': app})
    server = cheroot.wsgi.WSGIServer((host, port), dispatcher)
    app.attr['server'] = server
    return server
