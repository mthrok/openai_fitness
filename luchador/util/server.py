"""WSGI compliant server module"""
from __future__ import absolute_import

__all__ = ['create_server']


def create_server(app, port=5000, host='0.0.0.0'):
    """Mount application on cheroot WSGI server

    Parameters
    ----------
    app
        WSGI compilant application object.
        See :py:func:`create_env_app` for example.

    Returns
    -------
    WSGIServer object
    """
    import cheroot.wsgi
    dispatcher = cheroot.wsgi.WSGIPathInfoDispatcher({'/': app})
    server = cheroot.wsgi.WSGIServer((host, port), dispatcher)
    return server
