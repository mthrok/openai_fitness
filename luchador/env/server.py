"""Module for running Environment on remote server

Server module is for running Environment as an application independant from
Agent program. Environment methods are called with API routes with identical
names and parameters are passed via POST method. Server can be created using
:py:func:`create_env_app` as following.

>>> import luchador.env
>>> env = luchador.env.get_env(name)(**args)
>>> app = luchador.env.server.create_env_app(env)
>>> server = luchador.env.server.create_server(app)
>>> server.start()

Server API
----------

+---------+-------------------------------------------------------------------+
|**Retrie Environment information running on server**                         |
+---------+-------------------------------------------------------------------+
|URL      |``/`` or ``/info``                                                 |
+---------+-------------------------------------------------------------------+
|Method   |``POST`` or ``GET``                                                |
+---------+-------------------------------------------------------------------+
|Success  |**Code**: 200                                                      |
|Response |                                                                   |
|         |**Content (JSON)**                                                 |
|         | ``environment``: ``"Environment representation"``                 |
+---------+-------------------------------------------------------------------+

+---------+-------------------------------------------------------------------+
|**Retrie the outcome from the last state transition**                        |
+---------+-------------------------------------------------------------------+
|URL      |``/outcome``                                                       |
+---------+-------------------------------------------------------------------+
|Method   |``POST`` or ``GET``                                                |
+---------+-------------------------------------------------------------------+
|Success  |**Code**: 200                                                      |
|Response |                                                                   |
|         |**Content (JSON)**:                                                |
|         |                                                                   |
|         |  ``observation``: ``"Observation string"``                        |
|         |                                                                   |
|         |  ``reward``: ``Number``                                           |
|         |                                                                   |
|         |  ``terminal``: ``Boolean``                                        |
|         |                                                                   |
|         |  ``state``: ``Dictionary containing environment state``           |
+---------+-------------------------------------------------------------------+

+---------+-------------------------------------------------------------------+
|**Retrie the number of valid actions in the environment**                    |
+---------+-------------------------------------------------------------------+
|URL      |``/n_actions``                                                     |
+---------+-------------------------------------------------------------------+
|Method   |``POST`` or ``GET``                                                |
+---------+-------------------------------------------------------------------+
|Success  |**Code**: 200                                                      |
|Response |                                                                   |
|         |**Content (JSON)**:                                                |
|         | ``n_actions``: ``integer``                                        |
+---------+-------------------------------------------------------------------+

+---------+-------------------------------------------------------------------+
|**Reset environment**                                                        |
+---------+-------------------------------------------------------------------+
|URL      |``/reset``                                                         |
+---------+-------------------------------------------------------------------+
|Method   |``POST`` or ``GET``                                                |
+---------+-------------------------------------------------------------------+
|Success  |See ``/outcome``                                                   |
|Response |                                                                   |
+---------+-------------------------------------------------------------------+

+-----------+-----------------------------------------------------------------+
|**Advance environment with one step**                                        |
+-----------+-----------------------------------------------------------------+
|URL        |``/step``                                                        |
+-----------+-----------------------------------------------------------------+
|Method     |``POST``                                                         |
+-----------+-----------------------------------------------------------------+
|Data Params|``action``: ``integer``                                          |
+-----------+-----------------------------------------------------------------+
|Success    |See ``/outcome``                                                 |
|Response   |                                                                 |
+-----------+-----------------------------------------------------------------+

+---------+-------------------------------------------------------------------+
|**Shut down this environment and stop the server**                           |
+---------+-------------------------------------------------------------------+
|URL      |``/kill``                                                          |
+---------+-------------------------------------------------------------------+
|Method   |``POST`` or ``GET``                                                |
+---------+-------------------------------------------------------------------+
|Success  |**Code**: 200                                                      |
|Response |                                                                   |
|         |**Content (JSON)**:                                                |
|         | ``result``: ``success`` or ``failed``                             |
+---------+-------------------------------------------------------------------+
"""
from __future__ import absolute_import

import json
import logging

import flask
from cherrypy import wsgiserver

import luchador.env

_LG = logging.getLogger(__name__)


def _jsonify(obj, code=200):
    return (json.dumps(obj), code, {'mimetype': 'application/json'})


def create_env_app(env):
    """Create Flask server for the given Environment

    See module documentation for the derail of API call

    Parameters
    ----------
    env : Environment
        Environment to run on server

    Returns
    -------
    Flask
        Flask application which run the given environment
    """
    app = flask.Flask(__name__)

    attr = {
        'outcome': None,
        'env': env,
        'server': None,
    }

    @app.route('/', methods=['POST', 'GET'])
    @app.route('/info', methods=['POST', 'GET'])
    def _env_info():  # pylint: disable=unused-variable
        return _jsonify({'environment': repr(env)})

    @app.route('/outcome', methods=['POST', 'GET'])
    def _return_outcome():
        return _jsonify(luchador.env.serialize_outcome(attr['outcome']))

    @app.route('/n_actions', methods=['POST', 'GET'])
    def _n_actions():
        return _jsonify({'n_actions': env.n_actions})

    @app.route('/reset', methods=['POST', 'GET'])
    def _reset():
        attr['outcome'] = env.reset()
        return _return_outcome()

    @app.route('/step', methods=['POST'])
    def _step():
        try:
            params = flask.request.get_json()
        except Exception:
            _LG.exception('Failed to parse parameter')
            return 'Failed to parse parameter', 400

        if 'action' not in params:
            return 'Missing parameter; "action"', 400

        attr['outcome'] = env.step(params['action'])
        return _return_outcome()

    @app.route('/kill', methods=['POST', 'GET'])
    def _kill():
        attr['server'].stop()
        result = 'failed' if attr['server'].ready else 'killed'
        return _jsonify({'result': result})

    app.attr = attr
    _reset()
    return app


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
    dispatcher = wsgiserver.WSGIPathInfoDispatcher({'/': app})
    server = wsgiserver.CherryPyWSGIServer((host, port), dispatcher)
    app.attr['server'] = server
    return server
